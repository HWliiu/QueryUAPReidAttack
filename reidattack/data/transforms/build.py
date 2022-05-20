import torchvision.transforms as T


def build_transforms(tfm_cfg, is_train=True):
    res = []

    # normalize
    # do_norm = tfm_cfg.NORM.ENABLED
    # norm_pixel_mean = tfm_cfg.NORM.PIXEL_MEAN
    # norm_pixel_std = tfm_cfg.NORM.PIXEL_STD

    if is_train:
        size_train = tfm_cfg.SIZE_TRAIN
        if size_train[0] > 0:
            res.append(
                T.Resize(
                    size_train[0] if len(size_train) == 1 else size_train,
                    interpolation=T.InterpolationMode.BILINEAR))

        res.append(T.ToTensor())

        # !normalize need realized in train phase
        # if do_norm:
        #     res.append(
        #         T.Normalize(mean=norm_pixel_mean, std=norm_pixel_std))
    else:
        size_test = tfm_cfg.SIZE_TEST
        if size_test[0] > 0:
            res.append(
                T.Resize(
                    size_test[0] if len(size_test) == 1 else size_test,
                    interpolation=T.InterpolationMode.BILINEAR))

        res.append(T.ToTensor())

        # # !normalize need realized in val phase
        # if do_norm:
        #     res.append(
        #         T.Normalize(mean=norm_pixel_mean, std=norm_pixel_std))

    return T.Compose(res)
