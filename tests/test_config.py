import matplotlib as mpl

from dataviz import config


def test_project_folders_are_inside_the_project():
    assert config.DATA_DIR.parent == config.PROJECT_ROOT
    assert config.RAW_DATA_PATH.parent.name == "raw"
    assert config.OUTPUT_DIR.parent == config.PROJECT_ROOT


def test_column_groups_add_up_to_all_columns():
    expected = {
        config.ID_COLUMN,
        config.DATE_COLUMN,
        *config.CATEGORICAL_COLUMNS,
        *config.NUMERIC_COLUMNS,
    }
    assert set(config.COLUMNS) == expected
    assert len(config.COLUMNS) == len(set(config.COLUMNS))


def test_plot_style_uses_the_project_palette():
    config.apply_plot_style()
    colors = [item["color"] for item in mpl.rcParams["axes.prop_cycle"]]
    assert colors == list(config.PALETTE)


def test_plot_style_hides_the_top_and_right_borders():
    config.apply_plot_style()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False