import numpy as np


def egoh5(h5, ndims=2, bindcenter=8, align=True, b1=2, b2=11, silent=False):
    """
    Return h5 in the center of frame of body part index. If align is true, rotate data so that vector from b2 to b1
    always points to the right.
    """

    if ndims < 2:
        raise ValueError('Cant ego align 1D data.')
    if ndims > 2:
        # print(txt_format.RED+'%iD data. Aligning with the first two dimensions.'%ndims+txt_format.END)
        pass
    nbparts = int(h5.shape[1] / ndims)
    h5 = h5.reshape((-1, nbparts, ndims))
    ginds = np.setdiff1d(np.arange(h5.shape[1]), bindcenter)
    egoh5 = h5[:,:,:2] - h5[:,[bindcenter for i in range(h5.shape[1])],:2]
    dir_arr = egoh5[:, b1] - egoh5[:, b2]
    egoh5 = egoh5[:, ginds]
    if not align:
        return egoh5
    dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=1)[:, np.newaxis]
    if not silent:
        for t in tqdm(range(egoh5.shape[0])):
            rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
            egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    elif silent:
        for t in range(egoh5.shape[0]):
            rot_mat = np.array([[dir_arr[t, 0], dir_arr[t, 1]], [-dir_arr[t, 1], dir_arr[t, 0]]])
            egoh5[t] = np.array(np.dot(egoh5[t], rot_mat.T))
    if ndims>2:
        outdims = h5[:,:,2:] - h5[:, [bindcenter for i in range(h5.shape[1])], 2:]
        outdims = outdims[:, ginds]
        if len(outdims.shape)==2:
            outdims = outdims[..., None]
        egoh5 = np.concatenate([egoh5, outdims], axis=2)
    egoh5 = egoh5.reshape((-1, (nbparts-1)*ndims))
    return egoh5

def print_d(d, num=0, indent='\t', child_thresh=50):
    print(indent * num, d.name)
    if type(d) is h5py._hl.group.Group or type(d) is h5py._hl.files.File:
        if len(list(d.keys())) > child_thresh:
            print(indent * (num + 1), '%i children..' % len(list(d.keys())))
            return
        for i in list(d.keys()):
            if type(d[i]) is h5py._hl.group.Group:
                print_d(d[i], num=num + 1, indent=indent, child_thresh=child_thresh)
            else:
                print(indent * (num + 1), i, d[i].shape)
    else:
        print(indent * num, d.name, d.shape)

