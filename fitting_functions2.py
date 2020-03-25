# The installation of Theano is a little broken:
import os
import warnings

os.environ["MKL_THREADING_LAYER"] = "GNU"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Data manipulation
import numpy as np
import theano.tensor as tt
from ramjet.photometric_database.tess_data_interface import TessDataInterface
from astroquery.mast import Catalogs
import corner

# Plotting packages
# %matplotlib notebook
# %matplotlib tk
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import Range1d, CustomJS
import colorcet as cc


# Fitting
from astropy.io import fits
import pymc3 as pm
import exoplanet as xo


def downloading_lightcurve(tic_id, sector):
    tess_data_interface = TessDataInterface()
    lightcurve_path = tess_data_interface.download_lightcurve(tic_id=tic_id, sector=sector,
                                                              save_directory='lightcurves')
    print('You\'re using: ', lightcurve_path)
    return lightcurve_path


def getting_days_and_flux(lightcurve_path):
    fits_file = lightcurve_path
    # fits.info(fits_file)
    with fits.open(fits_file, mode="readonly") as hdulist:
        tess_bjds = hdulist[1].data['TIME']
        sap_fluxes = hdulist[1].data['SAP_FLUX']
        sap_fluxes_err = hdulist[1].data['SAP_FLUX_ERR']
        pdcsap_fluxes = hdulist[1].data['PDCSAP_FLUX']
        pdcsap_fluxes_err = hdulist[1].data['PDCSAP_FLUX_ERR']

    days = np.array(tess_bjds, dtype=np.float64)
    flux = np.array(pdcsap_fluxes, dtype=np.float64)

    nan_indexes = np.union1d(np.argwhere(np.isnan(flux)), np.argwhere(np.isnan(days)))
    flux = np.delete(flux, nan_indexes)
    days = np.delete(days, nan_indexes)

    mean_y = np.mean(flux)
    flux = ((flux / mean_y) - 1)
    return days, flux


def getting_star_mass_radius(tic_id):
    target_info = Catalogs.query_criteria(catalog='TIC', ID=tic_id).to_pandas()
    star_radius = target_info['rad']
    star_mass = target_info['mass']
    print(f'Star mass: {star_mass}')
    print(f'Star radius: {star_radius}')
    return star_mass, star_radius


def plotting_lightcurve(tic_id, sector, days, flux):
    fig = plt.figure(figsize=(6, 3))
    plt.scatter(days, flux, c=days, cmap='plasma', s=1)
    plt.xlabel("Time [days]")
    plt.ylabel("Relative flux [percantage]")
    plt.title(f'TIC {tic_id} Sector {sector}')
    pos = []

    def onclick(event):
        pos.append([event.xdata, event.ydata])

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return pos


def plotting_bokeh_click(tic_id, sector, flux,  days):
    callback = CustomJS(code="""
    // the event that triggered the callback is cb_obj:
    // The event type determines the relevant attributes
    console.log('Tap event occurred at x-position: ' + cb_obj.x)
    """)

    tools = "tap,pan,wheel_zoom,box_zoom,reset,previewsave"
    plot = figure(title=f"TIC {tic_id} Sector {sector} . Select your t0", tools=tools)

    number_for_dots = len(days) // len(cc.bmy) + 1
    palette = [cc.bmy[i // number_for_dots] for i in range(len(days))]
    plot.xaxis.axis_label = 'Time Since Transit [days]'
    plot.yaxis.axis_label = 'Relative Flux [percentage]'
    plot.circle(x=days, y=flux, line_color=None,
                fill_color=palette, fill_alpha=0.5, size=10)
    # execute a callback whenever the plot canvas is tapped
    plot.js_on_event('tap', callback)
    show(plot)

def folding(days, t0_guess, period_guess):
    x_fold = (days - t0_guess + 0.5 * period_guess) % period_guess - 0.5 * period_guess
    return x_fold


def plotting_folded(days, flux, tic_id, sector, x_fold):
    number_for_dots = len(days) // len(cc.bmy) + 1
    palette = [cc.bmy[i // number_for_dots] for i in range(len(days))]

    p = figure(title=f"TIC {tic_id} Sector {sector}")
    p.xaxis.axis_label = 'Time Since Transit [days]'
    p.yaxis.axis_label = 'Relative Flux [percentage]'
    p.circle(x_fold, flux, line_color=None,
             fill_color=palette, fill_alpha=0.5, size=10)

    # output_file("test.html", title="test.py example")
    return p


def manual_fitting(tic_id, sector, days, flux, t0_guess, period_guess, star_radius, star_mass, x_fold):
    with pm.Model() as model:
        lower_log = np.log(np.std(flux)) - 1
        logs = pm.Uniform("logs", lower=lower_log, upper=0,
                          testval=np.log(np.std(flux)))
        mean_flux = pm.Normal("mean_flux", mu=0, sd=np.std(flux))
        u = xo.distributions.QuadLimbDark("u")
        period = pm.Uniform("period",
                            lower=period_guess * 0.9,
                            upper=period_guess * 1.1,
                            testval=period_guess)

        t0 = pm.Uniform("t0", lower=t0_guess - 0.2, upper=t0_guess + 0.2)

        r, b = xo.distributions.get_joint_radius_impact(
            min_radius=0.0005, max_radius=0.5, testval_r=0.015)
        duration = pm.Uniform("duration", lower=0.05, upper=1.0)
        orbit = xo.orbits.KeplerianOrbit(
            period=period, t0=t0, r_star=star_radius, m_star=star_mass, duration=duration)

        # The light curve model is computed using "starry"
        star = xo.StarryLightCurve(u)
        light_curve = star.get_light_curve(
            orbit=orbit, r=r, t=days)
        # The returned light curve will always have the shape (ntime, nplanet)
        # but we only have one planet so we can "squeeze" the result
        # 1e2 it is because it's the percentage.

        light_curve = tt.squeeze(star.get_light_curve(orbit=orbit, r=r, t=days)) * 1e2 + mean_flux

        # Finally, this is the likelihoood for the observations
        pm.Normal("obs", mu=light_curve, sd=tt.exp(logs), observed=flux)
        with model:
            transit_model = xo.utils.eval_in_model(light_curve)
        inds = np.argsort(x_fold)

        p = plotting_folded(days, flux, tic_id, sector, x_fold)

        p.line(x_fold[inds], transit_model[inds], legend="initial model", line_width=3, line_alpha=0.6,
               line_color="black")

        # output_file("test.html", title="test.py example")

        show(p)
    return model, light_curve


def best_model(model):
    sampler = xo.PyMC3Sampler(finish=200)
    with model:
        sampler.tune(tune=2000, step_kwargs=dict(target_accept=0.9))
        trace = sampler.sample(draws=2000)
    return trace


def visualizing_fitting_process(days, trace, visual='yes'):
    varnames = ["period", "t0", "r", "duration"]
    if visual == 'yes':
        pm.traceplot(trace, varnames=varnames)
    labels = ["period [days]", "transit time [days]", "radius ratio"]
    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    if visual == 'yes':
        corner.corner(samples[["period", "t0", "r__0"]], labels=labels)
    # Compute the posterior parameters
    median_radius_ratio = np.median(trace["r"])
    median_impact_parameter = np.median(trace["b"])
    median_period = np.median(trace["period"])
    median_t0 = np.median(trace["t0"])
    median_x_fold = (days - median_t0 + 0.5 * median_period) % median_period - 0.5 * median_period
    median_inds = np.argsort(median_x_fold)
    return median_x_fold, median_t0, median_period, median_radius_ratio, median_impact_parameter


def plotting_final_fit(tic_id, sector, days, flux, trace, model, light_curve, median_x_fold):
    # Plot the data
    p = plotting_folded(days, flux, tic_id, sector, median_x_fold)
    # This is a little convoluted, but we'll take 100 random samples from the chain
    # and for each sample, we'll evaluate the predicted transit model and overplot it
    with model:
        # Pre-compile a function to evaluate the light curve
        func = xo.utils.get_theano_function_for_var(light_curve)

        # Loop over 100 random samples
        for sample in xo.utils.get_samples_from_trace(trace, size=100):
            # Fold the times based on the period and phase of this sample
            fold = (days - sample["t0"] + 0.5 * sample["period"]) % sample["period"] - 0.5 * sample["period"]
            inds = np.argsort(fold)

            # Evaluate the light curve
            args = xo.utils.get_args_for_theano_function(sample)
            transit_model = func(*args)

            # And plot the light curve model
            p.line(fold[inds], transit_model[inds], legend="optimized model", line_width=1, line_alpha=0.6,
                   line_color="black")
            p.x_range = Range1d(-0.4, 0.4)
    # Format the plot
    show(p)


def guidance(tic_id, sector):
    lightcurve_path = downloading_lightcurve(tic_id, sector)
    days, flux = getting_days_and_flux(lightcurve_path)
    star_mass, star_radius = getting_star_mass_radius(tic_id)
    plotting_bokeh_click(tic_id, sector, flux, days)
    # BUG HERE, the selection
    # pos = plotting_lightcurve(tic_id, sector, days, flux)
    # t0_guess = pos[-2][0]
    # next_t0_guess = pos[-1][0]
    # period_guess = next_t0_guess - t0_guess
    # print(f'Your t0 guess is: {t0_guess}')
    # print(f'Your next guess is: {next_t0_guess}')
    # print(f'Your period guess is: {period_guess}')
    # plt.switch_backend('Agg')
    t0_guess = 1441.6
    period_guess = 6.86
    print(f'Your t0 guess is: {t0_guess}')
    print(f'Your period guess is: {period_guess}')

    x_fold = folding(days, t0_guess, period_guess)
    p = plotting_folded(days, flux, tic_id, sector, x_fold)
    p.x_range = Range1d(-0.4, 0.4)
    show(p)
    model, light_curve = manual_fitting(tic_id, sector, days, flux, t0_guess, period_guess, star_radius, star_mass,
                                        x_fold)
    trace = best_model(model)
    pm.summary(trace, round_to=5)
    median_x_fold, median_t0, median_period, median_radius_ratio, median_impact_parameter = visualizing_fitting_process(
        days, trace, visual='no')
    print(f't0 = {median_t0}')
    print(f'period = {median_period}')
    print(f'radius_ratio = {median_radius_ratio}')
    print(f'impact_parameter = {median_impact_parameter}')
    plotting_final_fit(tic_id, sector, days, flux, trace, model, light_curve, median_x_fold)
    print("ooo")


if __name__ == '__main__':
    TIC_ID = 362043085
    SECTOR = 5
    guidance(TIC_ID, SECTOR)
