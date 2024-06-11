from contextlib import contextmanager
from typing import *
@contextmanager
def pd_display(
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        max_colwidth: Optional[int] = None,
        vertical_align: str = 'top',
        text_align: str = 'left',
        ignore_css: bool = False,
):
    try:
        from IPython.display import display
    except ImportError:
        display = print

    def disp(df: pd.DataFrame):
        css = [
            ## Align header to center
            {
                'selector': 'th',
                'props': [
                    ('vertical-align', 'center'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                ]
            },
            ## Align cell to top and left
            {
                'selector': 'td',
                'props': [
                    ('vertical-align', vertical_align),
                    ('text-align', text_align),
                    ('padding', '10px'),
                ]
            },

        ]
        if not ignore_css and isinstance(df, pd.DataFrame):
            df = df.style.set_table_styles(css)
        display(df)

    with pd.option_context(
            'display.max_rows', max_rows,
            'display.max_columns', max_cols,
            'max_colwidth', max_colwidth,
            'display.expand_frame_repr', False,
    ):
        yield disp
			