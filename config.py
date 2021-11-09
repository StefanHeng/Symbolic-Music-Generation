from icecream import ic


config = dict(
    datasets=dict(
        LMD_matched=dict(
            dir_nm='LMD-matched'
        )
    )
)

if __name__ == '__main__':
    import json
    from data_path import *

    fl_nm = 'config.json'
    ic(config)
    with open(f'{PATH_BASE}/{fl_nm}', 'w') as f:
        json.dump(config, f, indent=4)

