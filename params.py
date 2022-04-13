def get_param_dict(dataset: str):
    if dataset == 'PCam':
        params = {
            'NA_H': 0.13,
            'M_obj': 10,
            'M_relay': 1,
            'd_pixel': 9.72E-6,
            'img_ext': '.tif',
            'width': 96,
            'height': 96,
        }
    elif dataset == 'BreaKHis4X':
        params = {
            'NA_H': 0.16,
            'M_obj': 4,
            'M_relay': 3.3,
            'd_pixel': 6.5E-6,
            'img_ext': '.png',
            'width': 700,
            'height': 460,
        }
    elif dataset == 'BreaKHis10X':
        params = {
            'NA_H': 0.40,
            'M_obj': 10,
            'M_relay': 3.3,
            'd_pixel': 6.5E-6,
            'img_ext': '.png',
            'width': 700,
            'height': 460,
        }
    elif dataset == 'BreaKHis20X':
        params = {
            'NA_H': 0.80,
            'M_obj': 20,
            'M_relay': 3.3,
            'd_pixel': 6.5E-6,
            'img_ext': '.png',
            'width': 700,
            'height': 460,
        }
    elif dataset == 'BreaKHis40X':
        params = {
            'NA_H': 1.40,
            'M_obj': 40,
            'M_relay': 3.3,
            'd_pixel': 6.5E-6,
            'img_ext': '.png',
            'width': 700,
            'height': 460,
        }
    elif dataset == 'BACH':
        params = {
            'NA_H': 0.30,
            'M_obj': 20,
            'M_relay': 0.5,
            'd_pixel': 3.2E-6,
            'img_ext': '.tif',
            'width': 2048,
            'height': 1536,
        }
    else:
        raise ValueError("Unrecognized dataset")

    return params
