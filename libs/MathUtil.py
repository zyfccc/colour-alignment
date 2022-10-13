import cv2
import numpy as np

def length(v):
    if v.ndim == 1:
        return np.sqrt(np.sum(v ** 2))
    elif v.ndim == 2:
        return np.array([length(a) for a in v])
    raise ValueError('A very specific bad thing happened.')


def dot_product(v, w):
    if v.ndim == 1:
        return np.sum(v * w)
    elif v.ndim == 2:
        return np.array([dot_product(a, b) for a, b in zip(v, w)])
    raise ValueError('A very specific bad thing happened.')


def angle(v, w):
    rad = angle_ori(v, w)
    return rad.mean()


def angle_ori(v, w):
    length_v = length(v)
    length_w = length(w)
    product = dot_product(v, w)
    product_length = length_v * length_w
    cosx = np.divide(product, product_length, out=np.zeros_like(
        product), where=product_length != 0)
    cosx = np.clip(cosx, -1.0, 1.0)
    rad = np.arccos(cosx) % (np.pi)
    return rad


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def rmse_ori(v, w):
    return np.sqrt(((v - w) ** 2))


def deltaE2000(source, target):
    source = np.reshape(source, [source.shape[0], 1, 3]).astype(np.float32) / 255.0
    target = np.reshape(target, [target.shape[0], 1, 3]).astype(np.float32) / 255.0
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    source = np.reshape(source, [-1, 3])
    target = np.reshape(target, [-1, 3])
    deltaE00 = calc_deltaE2000(source, target)
    return deltaE00.mean()


def calc_deltaE2000(Labstd, Labsample):
    kl = 1
    kc = 1
    kh = 1
    Lstd = np.transpose(Labstd[:, 0])
    astd = np.transpose(Labstd[:, 1])
    bstd = np.transpose(Labstd[:, 2])
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    Lsample = np.transpose(Labsample[:, 0])
    asample = np.transpose(Labsample[:, 1])
    bsample = np.transpose(Labsample[:, 2])
    Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    Cabarithmean = (Cabstd + Cabsample) / 2
    G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
        Cabarithmean, 7) + np.power(25, 7))))
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
    Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
    Cpprod = (Cpsample * Cpstd)
    zcidx = np.argwhere(Cpprod == 0)
    hpstd = np.arctan2(bstd, apstd)
    hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
    dL = (Lsample - Lstd)
    dC = (Cpsample - Cpstd)
    dhp = (hpsample - hpstd)
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
    dhp[zcidx] = 0
    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    hp = (hpstd + hpsample) / 2
    hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
    Lpm502 = np.power((Lp - 50), 2)
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
        0.32 * np.cos(3 * hp + np.pi / 30) \
        - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (30 * np.pi / 180) * np.exp(
        - np.power((180 / np.pi * hp - 275) / 25, 2))
    Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
    RT = - np.sin(2 * delthetarad) * Rc
    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh
    de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                   np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
    return de00
