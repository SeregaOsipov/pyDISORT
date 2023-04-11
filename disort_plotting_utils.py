import math
from matplotlib import pyplot as plt, colors as colors
from climpy.utils.plotting_utils import JGR_page_width_inches, save_figure


def plot_spectral_profiles(ds, keys, yscale='log', xscale='log', yincrease=False, grid=False):
    ncols = math.ceil(len(keys)/3)
    fig, axes = plt.subplots(nrows=3, ncols=ncols, constrained_layout=True, figsize=(JGR_page_width_inches()*ncols/2, 3/2*JGR_page_width_inches()))
    indexer = 0
    for var_key in keys:
        var_ds = ds[var_key]
        ax = axes.flatten()[indexer]

        if grid:
            ax.grid()

        norm = None
        y_coord = None
        if 'level' in var_ds.dims:
            y_coord = 'level'
        if 'level_rho' in var_ds.dims:
            y_coord = 'level_rho'

        if var_ds.ndim > 1 and var_ds.quantile(0.05) > 0 and (var_key == 'od' or var_key == 'ssa'):
            norm = colors.LogNorm(vmin=var_ds.min(), vmax=var_ds.max())
            norm = colors.LogNorm(vmin=var_ds.quantile(0.05), vmax=var_ds.quantile(0.95))
            var_ds.plot(ax=ax, y=y_coord, yincrease=yincrease, norm=norm, xscale=xscale, yscale=yscale)  #
        else:
            var_ds.plot(ax=ax, y=y_coord, yincrease=yincrease, xscale=xscale, yscale=yscale)  #

        # ax.invert_yaxis()
        # ax.set_yscale('log')
        # if var_ds.ndim > 1:
        #     ax.set_xscale('log')

        print('{}: Min is {}, Max is {}'.format(var_key, var_ds.min().item(), var_ds.max().item()))
        indexer += 1

    return axes


def plot_ref_perturbed_pmc(wn_range, wn_range_label, perturbed_sim_label, disort_output_ds_ref, disort_output_ds_perturbed, disort_keys, model_label, pics_folder):
    ds = disort_output_ds_ref.sel(wavenumber=wn_range)
    axes1 = plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys, xscale='linear')
    plt.suptitle('{}. {}\nRef'.format(model_label, wn_range_label))
    save_figure(pics_folder, 'profiles_disort_ref_{}'.format(wn_range_label))

    ds = disort_output_ds_perturbed.sel(wavenumber=wn_range)
    axes2 = plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys, xscale='linear')
    plt.suptitle('{}. {}\n{}'.format(model_label, wn_range_label, perturbed_sim_label))
    save_figure(pics_folder, 'profiles_disort_{}_{}'.format(perturbed_sim_label.lower(), wn_range_label))

    ds = disort_output_ds_perturbed - disort_output_ds_ref
    ds = ds.sel(wavenumber=wn_range)
    axes3 = plot_spectral_profiles(ds.integrate('wavenumber'), disort_keys, xscale='linear')
    plt.suptitle('{}. {}\n{}-Ref'.format(model_label, wn_range_label, perturbed_sim_label))
    save_figure(pics_folder, 'profiles_disort_pmc_{}'.format(wn_range_label))
    return axes1, axes2, axes3
