# results dir and methods
import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .utils import DTO
import shutil
from .utils import is_pareto_efficient, mkdirs

import numpy as np

def retrive_results(
    dataset,
    log_dir="results"
):
    """retrive loaded results of a dataset from files

    Args:
        dataset (str): dataset name, e.g. Moji, Bios_both, and Bios_gender
        log_dir (str, optional): _description_. Defaults to "results".

    Returns:
        dict: experimental result dataframes of different methods.
    """
    log_dir = Path(log_dir)
    results = {}
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith(dataset):
                pre_len = len(dataset)+len(str(log_dir))+2
                file_path = os.path.join(root, file)
                if file.endswith("_df.pkl"):
                    mehtod = str(os.path.join(root, file))[pre_len:-7]
                else:
                    mehtod = str(os.path.join(root, file))[pre_len:-4]
                try:
                    results[mehtod] = pd.read_pickle(file_path)
                except:
                    import pickle5
                    with open(file_path, "rb") as fh:
                        results[mehtod] = pickle5.load(fh)
    return results


def final_results_df(
    results_dict,
    model_order = None,
    Fairness_metric_name = "fairness",
    Performance_metric_name = "performance",
    pareto = True,
    pareto_selection = "test",
    selection_criterion = "DTO",
    return_dev = True,
    Fairness_threshold = 0.0,
    Performance_threshold = 0.0,
    return_conf = False,
    save_conf_dir = None,
    num_trail = None,
    additional_metrics = [],
    ):
    """Process the results to a single dataset from creating tables and plots.

    Args:
        results_dict (dict): retrived results dictionary, which is typically the returned dict from function `retrive_results`
        model_order (list, optional): a list of models that will be considered in the final df. Defaults to None.
        Fairness_metric_name (str, optional): the metric name for fairness evaluation. Defaults to "fairness".
        Performance_metric_name (str, optional): the metric name for performance evaluation. Defaults to "performance".
        pareto (bool, optional): whether or not to return only the Pareto frontiers. Defaults to True.
        pareto_selection (str, optional): which split is used to select the frontiers. Defaults to "test".
        selection_criterion (str, optional): model selection criteria, one of {performance, fairness, both (DTO)} . Defaults to "DTO".
        return_dev (bool, optional): whether or not to return dev results in the df. Defaults to True.
        Fairness_threshold (float, optional): filtering rows with a minimal fairness as the threshold. Defaults to 0.0.
        Performance_threshold (float, optional): filtering rows with a minimal performance as the threshold. Defaults to 0.0.
        return_conf (bool, optional): return the selected epoch and corresponding YAML configure files if True. Defaults to False.
        save_conf_dir (str, optional): save selected epoch and configure files to the dir. Defaults to None.
        num_trail (int, optional): downsampling the number of searches of each method to $num_trail if not None. Defaults to None.
        additional_metrics (list, optional): report additional evaluation metrics for the selected epoch. Defaults to [].

    Returns:
        pandas.DataFrame: selected results of different models for report
    """

    df_list = []
    for key in (results_dict.keys() if model_order is None else model_order):
        _df = results_dict[key]

        # Calculate Mean and Variance for each run
        agg_dict = {
            "dev_performance":["mean", "std"],
            "dev_fairness":["mean", "std"],
            "test_performance":["mean", "std"],
            "test_fairness":["mean", "std"],
            "epoch":list,
            "opt_dir":list,
            }
        # Add aggregation to the specified metrics
        for _additional_metric in additional_metrics:
            agg_dict[_additional_metric] = ["mean", "std"]

        try:
            _df = _df.groupby(_df.index).agg(agg_dict).reset_index()
        except:
            print(key)
            break

        _df.columns = [' '.join(col).strip() for col in _df.columns.values]

        if num_trail is not None:
            _df = _df.sample(n=min(int(num_trail), len(_df)), random_state=1)

        # Select Pareto Frontiers
        _pareto_flag = is_pareto_efficient(
                -1*_df[["{}_{} mean".format(pareto_selection, Fairness_metric_name), "{}_{} mean".format(pareto_selection, Performance_metric_name)]].to_numpy()
                )
        _df["is_pareto"] = _pareto_flag
        if pareto:
            _pareto_df = _df[_pareto_flag].copy()
        else:
            _pareto_df = _df.copy()

        # Filtering based on min fairness and performance
        _tmp_df = _pareto_df[_pareto_df["dev_{} mean".format(Performance_metric_name)]>=Performance_threshold].copy()
        _tmp_df = _tmp_df[_tmp_df["dev_{} mean".format(Fairness_metric_name)]>=Fairness_threshold].copy()

        if len(_tmp_df) >= 1:
            _pareto_df = _tmp_df
        
        # Rename and reorder the columns
        selected_columns = ["{}_{} {}".format(phase, metric, value) for phase in ["test", "dev"] for metric in [Performance_metric_name, Fairness_metric_name] for value in ["mean", "std"]]
        selected_columns.append("epoch list")
        selected_columns.append("opt_dir list")
        selected_columns.append("is_pareto")
        # Consider the selected models
        additional_metric_columns = ["{} {}".format(metric, value) for metric in additional_metrics for value in ["mean", "std"]]
        selected_columns = selected_columns + additional_metric_columns

        _pareto_df = _pareto_df[selected_columns].copy()
        _pareto_df["Models"] = [key]*len(_pareto_df)

        _final_DTO = DTO(
            fairness_metric=list(_pareto_df["dev_{} mean".format(Fairness_metric_name)]), 
            performacne_metric=list(_pareto_df["dev_{} mean".format(Performance_metric_name)]),
            utopia_fairness = 1, utopia_performance = 1
            )
        _pareto_df["dev_DTO mean"] = _final_DTO

        # Model selection
        if selection_criterion is not None:
            if selection_criterion == "DTO":
                selected_epoch_id = np.argmin(_pareto_df["dev_{} mean".format(selection_criterion)])
            else:
                selected_epoch_id = np.argmax(_pareto_df["dev_{} mean".format(selection_criterion)])
            _pareto_df = _pareto_df.iloc[[selected_epoch_id]].copy()
        
        df_list.append(_pareto_df)

    final_df = pd.concat(df_list)
    final_df.reset_index(inplace=True)

    if selection_criterion is not None:
        _over_DTO = DTO(
            fairness_metric=list(final_df["test_{} mean".format(Fairness_metric_name)]), 
            performacne_metric=list(final_df["test_{} mean".format(Performance_metric_name)]),
            utopia_fairness = 1, utopia_performance = 1
            )
        final_df["DTO"] = _over_DTO

        if save_conf_dir is not None:
            for (_model, _epoch_list, opt_list) in final_df[["Models", "epoch list", "opt_dir list"]].values:
                model_dir = Path(save_conf_dir)/_model
                mkdirs(model_dir)
                for _run, (_epoch, _opt) in enumerate(zip(_epoch_list, opt_list)):
                    shutil.copy2(_opt, model_dir / 'Run_{}_Selected_Epoch_{}.yaml'.format(_run, _epoch))

        evaluation_cols = list(final_df.keys())[1:(9 if return_dev else 5)]
        reproducibility_cols = ["epoch list", "opt_dir list"] if return_conf else []
        final_df = final_df[["Models"]+evaluation_cols+["DTO"]+reproducibility_cols+["is_pareto"]+additional_metric_columns].copy()

    return final_df

def interactive_plot(plot_df, figsize=(12, 7), dpi = 100, selection="DTO"):
    """Create interactive plots for DTO and constrained selection.

    Args:
        plot_df (_type_): a pd.DataFrame including numbers for each method.
        figsize (tuple, optional): figure size in tuple. Defaults to (12, 7).
        dpi (int, optional): figure resolution. Defaults to 100.
        selection (str, optional): constrained | DTO, indicating which model selection approach is used. Defaults to "DTO".
    """
    from matplotlib.widgets import CheckButtons
    from matplotlib.widgets import Slider, Button, RangeSlider

    assert selection in ["constrained", "DTO"], NotImplementedError

    plot_df["Fairness"] = plot_df["test_fairness mean"]
    plot_df["Performance"] = plot_df["test_performance mean"]

    fig, ax = plt.subplots(2, 3, figsize=figsize, dpi = dpi, gridspec_kw={'width_ratios': [1, 0.2, 2.5], 'height_ratios': [1, 0.075]})

    with sns.axes_style("white"):
        plot_ax = sns.lineplot(
            data=plot_df,
            x="Performance",
            y="Fairness",
            hue="Models",
            markers=True,
            style="Models",
            ax = ax[0, 2]
        )

    plot_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    init_performance_max = max(plot_df["Performance"])
    init_fairness_max = max(plot_df["Fairness"])

    init_performance_min = min(plot_df["Performance"])
    init_fairness_min = min(plot_df["Fairness"])

    ax[0, 2].set_xlim(
        1.1*init_performance_min-0.1*init_performance_max, 
        1.1*init_performance_max-0.1*init_performance_min)

    ax[0, 2].set_ylim(
        1.1*init_fairness_min-0.1*init_fairness_max, 
        1.1*init_fairness_max-0.1*init_fairness_min)


    # Selection box for displaying different models
    lines = list(plot_ax.lines)
    num_models = int(len(lines)/2)

    # Make checkbuttons with all plotted lines with correct visibility
    labels = [str(line.get_label()) for line in lines[num_models:]]
    visibility = [line.get_visible() for line in lines[num_models:]]
    # check = CheckButtons(rax, labels, visibility)
    check = CheckButtons(ax[0, 0], labels, visibility)

    def func(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    check.on_clicked(func)

    # Vary the lines
    if selection == "constrained":
        performance_limit_line = ax[0, 2].axvline(init_performance_max, ls="--", color='r') # vertical
        fairness_limit_line = ax[0, 2].axhline(init_fairness_max, ls="--", color='r') # horizontal
    elif selection == "DTO":
        performance_limit_line = ax[0, 2].axvline(init_performance_max, ls="--", color='g') # vertical
        fairness_limit_line = ax[0, 2].axhline(init_fairness_max, ls="--", color='g') # horizontal

    fairness_slider = RangeSlider(
        ax=ax[0, 1],
        label="Fairness",
        valmin=0,
        valmax=1,
        valinit=(1.1*init_fairness_min-0.1*init_fairness_max, init_fairness_max),
        orientation="vertical"
    )

    performacne_slider = RangeSlider(
        ax=ax[1, 2],
        label="Performance",
        valmin=0,
        valmax=1,
        valinit=(1.1*init_performance_min-0.1*init_performance_max, init_performance_max),
    )

    def performance_update(val):
        performance_limit_line.set_xdata(val[1])

        ax[0, 2].set_xlim(
            1.1*val[0]-0.1*val[1], 
            1.1*val[1]-0.1*val[0])

        fig.canvas.draw_idle()

    def fairness_update(val):
        fairness_limit_line.set_ydata(val[1])

        ax[0, 2].set_ylim(
            1.1*val[0]-0.1*val[1], 
            1.1*val[1]-0.1*val[0])

        fig.canvas.draw_idle()

    performacne_slider.on_changed(performance_update)
    fairness_slider.on_changed(fairness_update)

    # Reset fairness and performacne line
    if selection == "constrained":
        tradeoff_reset_button = Button(ax[1,1], 'RES', hovercolor='0.975')
    elif selection == "DTO":
        tradeoff_reset_button = Button(ax[1,1], 'DTO', hovercolor='0.975')
        utopia_point = ax[0, 2].plot(
            performacne_slider.val[1], 
            fairness_slider.val[1], 
            "go")[0]

        utopia_label = ax[0, 2].text(
            performacne_slider.val[1], 
            fairness_slider.val[1], 
            "Utopia", c="k", horizontalalignment='left',
            verticalalignment='bottom', fontsize=12)

        utopia_point.set_visible(False)
        utopia_label.set_visible(False)

    if selection == "constrained":
        def reset(event):
            performacne_slider.reset()
            fairness_slider.reset()

            ax[0, 2].set_xlim(
                1.1*init_performance_min-0.1*init_performance_max, 
                1.1*init_performance_max-0.1*init_performance_min)

            ax[0, 2].set_ylim(
                1.1*init_fairness_min-0.1*init_fairness_max, 
                1.1*init_fairness_max-0.1*init_fairness_min)
    elif selection == "DTO":
        # Handle the click event
        global DTO_lines
        DTO_lines = []
        global DTO_line_labels
        DTO_line_labels = []

        def DTO_onclick(event):
            if event.inaxes not in [ax[0, 2]]:
                return
            _xdata, _ydata = event.xdata, event.ydata
            # Draw a line between 
            _line = ax[0, 2].plot(
                [_xdata, performacne_slider.val[1]], 
                [_ydata, fairness_slider.val[1]], 'g:')
            DTO_lines.append(_line[0])

            # Calculate and display DTO
            _DTO = DTO(
                fairness_metric = [_ydata], 
                performacne_metric = [_xdata], 
                utopia_fairness = fairness_slider.val[1], 
                utopia_performance = performacne_slider.val[1])
            _DTO_label = ax[0, 2].text(
            _xdata, _ydata, 
            "{:.4f}".format(_DTO[0]), c="k", fontsize=9, 
            # horizontalalignment='left', verticalalignment='bottom',
            )
            DTO_line_labels.append(_DTO_label)

        global DTO_pick_cid
        DTO_pick_cid = fig.canvas.mpl_connect('button_press_event', DTO_onclick)
        fig.canvas.mpl_disconnect(DTO_pick_cid)


        def reset(event):
            global DTO_pick_cid
            global DTO_lines
            global DTO_line_labels

            if tradeoff_reset_button.label.get_text() == "DTO":
                # Change the button label
                tradeoff_reset_button.label.set_text("RES")
                # Update the location of Utopoia point
                utopia_point.set_xdata(performacne_slider.val[1])
                utopia_label.set_x(performacne_slider.val[1])
                utopia_point.set_ydata(fairness_slider.val[1])
                utopia_label.set_y(fairness_slider.val[1])
                # Display the Utopia point
                utopia_point.set_visible(True)
                utopia_label.set_visible(True)
                # Connect the pick event handler
                DTO_pick_cid = fig.canvas.mpl_connect('button_press_event', DTO_onclick)

            else:
                # Change the button label
                tradeoff_reset_button.label.set_text("DTO")

                performacne_slider.reset()
                fairness_slider.reset()

                ax[0, 2].set_xlim(
                    1.1*init_performance_min-0.1*init_performance_max, 
                    1.1*init_performance_max-0.1*init_performance_min)

                ax[0, 2].set_ylim(
                    1.1*init_fairness_min-0.1*init_fairness_max, 
                    1.1*init_fairness_max-0.1*init_fairness_min)
                
                # Remove utopia point
                utopia_point.set_visible(False)
                utopia_label.set_visible(False)

                # Disconnet the pick event handler
                fig.canvas.mpl_disconnect(DTO_pick_cid)
                # Remove Lines
                for line in DTO_lines:
                    # line.set_visible(False)
                    line.remove()
                DTO_lines = []
                # Remove labels
                for label in DTO_line_labels:
                    label.remove()
                DTO_line_labels = []
            
            fig.canvas.draw_idle()

    tradeoff_reset_button.on_clicked(reset)

    # ax[1,0].axis('off')
    method_selectAll_button = Button(ax[1,0], 'Unselect', hovercolor='0.975')
    def select_all_methods(event):
        if method_selectAll_button.label.get_text() == "Select All":
            method_selectAll_button.label.set_text("Unselect")

            for index in range(num_models):
                if not lines[index].get_visible():
                    # lines[index].set_visible(True)
                    check.set_active(index)
            plt.draw()

        else:
            method_selectAll_button.label.set_text("Select All")

            for index in range(num_models):
                if lines[index].get_visible():
                    # lines[index].set_visible(False)
                    check.set_active(index)
            plt.draw()
    
    method_selectAll_button.on_clicked(select_all_methods)



    plt.tight_layout()
    # plt.subplots_adjust(right=0.75)  
    plt.show()

def make_zoom_plot(
    plot_df, figure_name=None,
    xlim=None, ylim=None,
    figsize=(7.5, 6), dpi = 150,
    zoom_xlim=None, zoom_ylim=None,
    zoomed_location = [1.05, 0.05, 0.37, 0.9]
    ):
    """Make tradeoff plots with zoomed-in area.

    Args:
        plot_df (pd.DataFrame): a pd.DataFrame including numbers for each method.
        figure_name (str, optional): save the plot with figure_name. Defaults to None.
        xlim (tuple, optional): x-axis limit. Defaults to None.
        ylim (tuple, optional): y-aix limit. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to (7.5, 6).
        dpi (int, optional): figure resolution. Defaults to 150.
        zoom_xlim (tuple, optional): x-axis interval of the zoomed-in area. Defaults to None.
        zoom_ylim (tuple, optional): y-axis interval of the zoomed-in area. Defaults to None.
        zoomed_location (list, optional): location of the zoomed-in area, [x, y, length, height]. Defaults to [1.05, 0.05, 0.37, 0.9].
    """

    plot_df["Fairness"] = plot_df["test_fairness mean"]
    plot_df["Accuracy"] = plot_df["test_performance mean"]

    # fig, ax = plt.subplots(1, 2, figsize=figsize, dpi = dpi, gridspec_kw={'width_ratios': [0.8, 0.2]})
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)

    with sns.axes_style("white"):
        sns.lineplot(
            data=plot_df,
            x="Accuracy",
            y="Fairness",
            hue="Models",
            markers=True,
            style="Models",
            ax=ax
        )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    sns.move_legend(ax, "lower left")

    axins = ax.inset_axes(zoomed_location)
    with sns.axes_style("white"):
        sns.lineplot(
            data=plot_df,
            x="Accuracy",
            y="Fairness",
            hue="Models",
            markers=True,
            style="Models",
            legend=False,
            ax=axins
        )

    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    
    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")
    
    if figure_name is not None:
        fig.savefig(figure_name+".pdf", format="pdf", dpi=960, bbox_inches="tight")