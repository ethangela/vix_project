import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from argparse import ArgumentParser
from skimage.io import imread, imsave
#import torch.nn as nn 
#import torch.optim as optim
import random

def kernel(x1, x2, l=0.5, sigma_f=0.2):
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

def sample_from_2D_Gaussian(size):
    m = size**2
    test_d1 = np.linspace(-10, 10, m)
    test_d2 = np.linspace(-10, 10, m)
    
    test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
    test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]
    test_X = np.asarray(test_X)
    mu = np.zeros_like(test_d1)
    cov = kernel(test_X, test_X)
    print('parameter set done')
    
    gp_samples = np.random.multivariate_normal(
            mean = mu.ravel(), 
            cov = cov, 
            size = 1)
    z = gp_samples.reshape(test_d1.shape)
    print('sampling done')
    
    #scale to range(0,1)
    z = (z - np.min(z))/np.ptp(z)
    np.save('2D.npy', z)
    #print(z)
    test_d1 = (test_d1 - np.min(test_d1))/np.ptp(test_d1)
    test_d2 = (test_d2 - np.min(test_d2))/np.ptp(test_d2)
    print('scaling done')
    
    fig = plt.figure(figsize=(5, 5))
    plt.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=1)
    #ax.set_title("with optimization l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
    #plt.show()
    plt.savefig('Jun25.jpg')
    
def excel(path):
    df = pd.read_pickle(path)
    csv_title = path[:-3] + 'csv'
    df.to_csv(csv_title,index=0)
    pass








def plot_lay(lay_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))
 
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 256) & (df['input_size'] == 768) & (df['filter_size'] == 4)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='layers')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(lay_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('layers')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('layers for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/layer/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/layer_for_{}_on_{}_signal.png'.format(mask_inf, type))

def plot_chn(chn_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path) 
    plt.figure(figsize=(16,10))
    
    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']    

    if mask_inf != 'circulant':
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='channels')
    y_avg = []
    for sgl in sgl_list:
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(chn_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr)
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(chn_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    
    
    plt.xlabel('channels')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('channels for {} on signal {}'.format(mask_inf, type))
    # show a legend on the plot
    plt.legend()
    
    path_dir = 'figures/channel/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/channel_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_ipt(ipt_list, type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy']#, '1D_rbf_3.0_4.npy', '1D_rbf_3.0_5.npy'] ##########

    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['channels'] == 6) & (df['filter_size'] == 15)] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='input_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(ipt_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(ipt_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('input_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('input_size for {} on signal {}'.format(mask_inf, type))    
    # show a legend on the plot
    plt.legend()

    path_dir = 'figures/input_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/input_size_for_{}_on_{}_signal.png'.format(mask_inf, type))




def plot_fit(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['layers'] == 1) & (df['input_size'] == 628) & (df['channels'] == 6  )] #a set of channels on 3 signals THREE TIMES?#########

    df_sel = df_sel.sort_values(by ='filter_size')#####
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        if len(psnr) > 10:
            p = []
            l = int(len(psnr)/3)
            for i in range(3):
                p.append(psnr[i*l:(i+1)*l])
            psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        plt.plot(fit_size, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(fit_size, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('filter_sizes')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('filter_size for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/filter_size/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/filter_size_for_{}_on_{}_signal.png'.format(mask_inf, type))





def plot_activation(type, csv_path, mask_inf='circulant'):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(16,10))

    if type == 'exp':
        sgl_list = ['1D_exp_0.25_6.npy', '1D_exp_0.25_7.npy', '1D_exp_0.25_8.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_6.npy', '1D_rbf_3.0_7.npy', '1D_rbf_3.0_8.npy'] 
    
    if mask_inf != 'circulant':
        #mask = df['mask_info'] == mask_inf
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10) & (df['mask_info'] == mask_inf)] #a set of channels on 3 signals######## 
    else:
        df_sel = df[(df['channels'] == 384) & (df['layers'] == 4) & (df['input_size'] == 384) & (df['filter_size'] == 10)] #a set of channels on 3 signals THREE TIMES?#########

    act_list = ['relu', 'leaky_relu', 'sigmoid']
    y_avg = []
    for sgl in sgl_list:
        #print(df_sel['img_name'])
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        plt.plot(act_list, psnr, '--', label = sgl, alpha=0.6)#####
        y_avg.append(psnr) 
    y_plt = [(x+y+z)/3 for x,y,z in zip(*y_avg)]
    plt.plot(act_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel('act_functions')#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    plt.title('act_function for {} on signal {}'.format(mask_inf, type)) 
    # show a legend on the plot
    plt.legend()
    path_dir = 'figures/single_layer/act_function/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass 
    else:
        os.makedirs(path_dir)
    plt.savefig(path_dir+'/act_function{}_on_{}_signal.png'.format(mask_inf, type))



def picture(path, mask=None, type='block'):
    plt.figure(figsize=(80,10))
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)
    fs = np.load(path)
    print('signal',fs.shape)
    title = path[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    
    if mask:
        m = np.load('mask/1D_mask_'+type+'_4096_'+str(mask)+'_1.npy')
        fs = fs * m
        fs[fs == 0.] = np.nan
        title += '_masked_'+type+'_'+str(mask)
    
    plt.plot(xs, fs, 'gray') # Plot the samples
    plt.savefig(tn + '.jpg')
    plt.close()


def group_pic(org, msk, path, title_m):
    org1, org2, org3 = org
    path1, path2, path3 = path

    plt.figure(figsize=(240,30))
    
    xs = np.linspace(-10,10,4096) #Range vector (101,)
    xs = (xs - np.min(xs))/np.ptp(xs)

    # original
    plt.subplot(331)
    fs = np.load(org1)
    plt.title(org1[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(334)
    fs = np.load(org2)
    plt.title(org2[:-4])
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(337)
    fs = np.load(org3)
    plt.title(org3[:-4])
    plt.plot(xs, fs, 'gray')


    # masked
    plt.subplot(332)
    m = np.load(msk)
    fs = np.load(org1)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(335)
    m = np.load(msk)
    fs = np.load(org2)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')
    
    plt.subplot(338)
    m = np.load(msk)
    fs = np.load(org3)
    fs = fs * m
    fs[fs == 0.] = np.nan
    plt.title(str(1/int(msk[-7])))
    plt.plot(xs, fs, 'gray')


    # recovered
    plt.subplot(333)
    fs = np.load(path1)
    title = path1[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(336)
    fs = np.load(path2)
    title = path2[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.subplot(339)
    fs = np.load(path3)
    title = path3[:-4]
    tc = title.split('_')
    tn = '_'.join(tc[:14])
    plt.title('filter_size: {}, #channels: {}, #layers: {}, #input_size: {}'.format(tc[17], tc[19], tc[21], tc[24]))
    plt.plot(xs, fs, 'gray')

    plt.savefig(title_m + '.jpg')
    print('saved')
    plt.close()


def ipt(hparams):
    o = 4096 / (pow(hparams.f, hparams.l))
    print(o)



def plot_jan(csv_path, mea_type, mea_range, sig_type, mask_inf):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8,5))
 
    if sig_type == 'exp':
        sgl_list = ['1D_exp_0.25_4096_10.npy', '1D_exp_0.25_4096_11.npy', '1D_exp_0.25_4096_12.npy', '1D_exp_0.25_4096_13.npy', '1D_exp_0.25_4096_14.npy']    
    else:
        sgl_list = ['1D_rbf_3.0_4096_10.npy', '1D_rbf_3.0_4096_11.npy', '1D_rbf_3.0_4096_12.npy', '1D_rbf_3.0_4096_13.npy', '1D_rbf_3.0_4096_14.npy'] 

    df_sel = df

    # if mea_type == 'layers':
    #     #lay_list = [1,2,3,4,5,6,7,8,9,10,12,16,20,25,30,35,40,50,60,70,85,100,120,140,180,240] #3,3,7
    # elif mea_type == 'channels':
    #     #lay_list = [50,80,120,140,160,180,200,220,260,300,350,400,450,500,600,700,850,1000]#160,180,160
    # elif mea_type == 'input_size':
    #     #lay_list = [30,50,70,80,90,95,100,105,110,120,130,140,150,180,200,240,330,384,420,480,520]#120,200,384

    lay_list = mea_range

    df_sel = df_sel.sort_values(by = mea_type)##### e.g.'layers'
    y_avg = []
    for i,sgl in enumerate(sgl_list):
        psnr = df_sel[df_sel['img_name'] == sgl]['psnr'].tolist()
        # if len(psnr) > 10:
        #     p = []
        #     l = int(len(psnr)/3)
        #     for i in range(3):
        #         p.append(psnr[i*l:(i+1)*l])
        #     psnr = [(x+y+z)/3 for x,y,z in zip(*p)]
        label_title = 'testing signal {}'.format(int(i+1))
        plt.plot(lay_list, psnr, '--', label = label_title, alpha=0.6)#####
        y_avg.append(psnr)
    
    y_plt = [(x+y+z+e+f)/5 for x,y,z,e,f in zip(*y_avg)]
    plt.plot(lay_list, y_plt, 'r.-', label = 'average', linewidth=3)#####    

    plt.xlabel(mea_type)#####
    # Set the y axis label of the current axis.
    plt.ylabel('PSNR')
    # Set a title of the current axes.
    mask_info_ls = mask_inf.split('_') #'random_8', 'block_2', 'denoise_0.05', 'circulant_100'
    if mask_info_ls[0] == 'random' or mask_info_ls[0] == 'block':
        mask_info = '1_{} of entries are {}-masked'.format(int(mask_info_ls[1]), mask_info_ls[0])
    elif mask_info_ls[0] == 'denoise':
        mask_info = 'medium noise added'
    elif mask_info_ls[0] == 'circulant':
        mask_info = 'compressing at ratio of {}_4096'.format(int(mask_info_ls[1]))
    plot_title =  mea_type + ' for '+ sig_type +' signals with measurement as '+ mask_info
    #plt.title(plot_title)
    # show a legend on the plot
    plt.legend()

    path_dir = 'feb6_hyper_figures/{}'.format(mask_inf)
    if os.path.exists(path_dir):
        pass
    else:
        os.makedirs(path_dir)
    img_name = plot_title +'.jpg'
    final = os.path.join(path_dir, img_name)
    plt.savefig(final)


def qplot(sig_type, plot_title, plot_path=None): 
    plt.figure(figsize=(14,4))
    xs = np.linspace(-10,10,4096) 
    xs = (xs - np.min(xs))/np.ptp(xs)
    if sig_type == 'exp':
        path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
    elif sig_type == 'rbf':
        path = 'Gaussian_signal/1D_rbf_3.0_4096_30.npy'
    elif sig_type == 'recover':
        path = 'result/' + plot_path
    fs = np.load(path)
    plt.plot(xs, fs, 'gray')
    plt.savefig('feb5_figures/{}.jpg'.format(plot_title))
    print('single plot saved')
    plt.close()

def oplot(sig_type, mea_path, plot_title): 
    plt.figure(figsize=(14,4))
    xs = np.linspace(-10,10,4096) 
    xs = (xs - np.min(xs))/np.ptp(xs)
    if sig_type == 'exp':
        path = 'Gaussian_signal/1D_exp_0.25_4096_30.npy'
    elif sig_type == 'rbf':
        path = 'Gaussian_signal/1D_rbf_3.0_4096_30.npy'
    fs = np.load(path)
    if mea_path != 'denoise':
        ms = np.load(mea_path)
        ms = ms.reshape(4096)
        ys = fs * ms
        ys[ys == 0.] = np.nan
    else:
        ys = fs + 0.05 * np.random.randn(4096)
    plt.plot(xs, ys, 'gray')
    plt.savefig('feb5_figures/{}.jpg'.format(plot_title))
    print('single plot saved')
    plt.close()


def main(hparams):
    if hparams.plot == 'arch':
        plot_jan(hparams.csv, 
        hparams.mea_type, 
        hparams.mea_range, 
        hparams.sig_type,
        hparams.mask_inf)
    elif hparams.plot == 'original':
        qplot(hparams.sig_type,
        hparams.title)
    elif hparams.plot == 'observe':
        oplot(hparams.sig_type,
        hparams.mea_path,
        hparams.title)
    elif hparams.plot == 'recover':
        qplot(hparams.sig_type,
        hparams.title,
        hparams.recover_path)

    print('ploting completed')



if __name__ == '__main__':    
    # PARSER = ArgumentParser()
    # PARSER.add_argument('--csv', type=str, default='inpaint_lay_rbf_block_14.pkl', help='path stroing pkl')
    # PARSER.add_argument('--mea_type', type=str, default='layers', help='layers/channels/input_size/filter_size/step_size')
    # PARSER.add_argument('--mea_range', type=float, nargs='+', default=[1,2,3,4,5,6,7,8,9,10,12,16,20], help='a list of xs')
    # PARSER.add_argument('--sig_type', type=str, default='exp', help='exp/rbf')
    # PARSER.add_argument('--mask_inf', type=str, default='random_8', help='mask info, e.g., block_2, denoise_0.05, circulant_100') 
    # PARSER.add_argument('--plot', type=str, default='arch') 
    # PARSER.add_argument('--mea_path', type=str, default='Masks/1D_mask_random_64_2_1.npy') 
    # PARSER.add_argument('--title', type=str, default='original_exp') 
    # PARSER.add_argument('--recover_path', type=str, default='recover_path') 
    
    # HPARAMS = PARSER.parse_args()
    
    # main(HPARAMS)

    # a = np.random.randint(5, size=(5,2))
    # print('a:')
    # print(a)
    # dim = 2
    # for axis in range(dim):
    #     print('axis is {}'.format(axis))
    #     zero_shape = list(a.shape) #[5,1]
    #     zero_shape[axis] = 1
    #     d1 = np.diff(a, axis=axis)
    #     print('d1:')
    #     print(d1)
    #     d2 = np.zeros(zero_shape)
    #     print('d2:')
    #     print(d2)
    #     diff = np.concatenate((d1,d2),axis=axis)
    #     print('diff:')
    #     print(diff)


    # def load_img(path, img_name):
    #     img_path = os.path.join(path, img_name)
    #     img = imread(img_path)
    #     img = np.transpose(img, (1, 0, 2))
    #     img = np.pad(img, ((23,23), (3,3), (0,0)), 'constant', constant_values=0)
    #     #img = img[None, :, :, :]
    #     img_clean = img / 255.
    #     return img_clean

    # img = load_img('Celeb_signal', '182649.jpg')
    # print(img.shape)

    # img = img * 255.
    # img = np.transpose(img, (1, 0, 2))
    # imsave(os.path.join('Celeb_signal', 'test.jpg'), img.astype(np.uint8))


    #generate basis
    #def generate_basis():
    #    """generate the basis"""
    #    x = np.zeros((224, 224)) ##########################!!!!!!!!!!!!!!!!!#########
    #    coefs = pywt.wavedec2(x, 'db1')
    #    n_levels = len(coefs)
    #    basis = []
    #    for i in range(n_levels):
    #        coefs[i] = list(coefs[i])
    #        n_filters = len(coefs[i])
    #        for j in range(n_filters):
    #            for m in range(coefs[i][j].shape[0]):
    #                try:
    #                    for n in range(coefs[i][j].shape[1]):
    #                        coefs[i][j][m][n] = 1
    #                        temp_basis = pywt.waverec2(coefs, 'db1')
    #                        basis.append(temp_basis)
    #                        coefs[i][j][m][n] = 0
    #                except IndexError:
    #                    coefs[i][j][m] = 1
    #                    temp_basis = pywt.waverec2(coefs, 'db1')
    #                    basis.append(temp_basis)
    #                    coefs[i][j][m] = 0
    #
    #    basis = np.array(basis)
    #    return basis
        
    #basis = generate_basis()
    #np.save('./wavelet_basis.npy', basis) #
    #print(basis.shape)
    predict= [25.778675079345703, 27.591697692871094, 27.491653442382812, 28.477962493896484, 28.427940368652344, 28.074291229248047, 27.966625213623047, 27.305221557617188, 27.36477279663086, 25.217315673828125, 24.734722137451172, 25.847911834716797, 25.367382049560547, 24.992454528808594, 25.61145782470703, 25.45376968383789, 24.753936767578125, 24.371387481689453, 24.390602111816406, 24.084911346435547, 23.844646453857422, 23.485122680664062, 23.021743774414062, 23.071765899658203, 22.781478881835938, 23.021743774414062, 22.869930267333984, 23.725387573242188, 23.79462432861328, 23.642810821533203, 23.588977813720703, 23.863861083984375, 24.35979461669922, 24.20417022705078, 24.901939392089844, 25.64226531982422, 24.892410278320312, 24.646270751953125, 24.371387481689453, 24.431095123291016, 26.10564422607422, 25.828697204589844, 25.680694580078125, 24.904003143310547, 24.50986099243164, 25.059627532958984, 24.884788513183594, 24.686763763427734, 27.88579559326172, 27.305221557617188, 27.355243682861328, 25.523006439208984, 24.034889221191406, 25.128864288330078, 25.04041290283203, 24.87319564819336, 25.947956085205078, 26.440078735351562, 26.055622100830078, 24.1041259765625, 23.69076919555664, 24.421409606933594, 26.29001235961914, 26.142009735107422, 29.89684295654297, 28.82208251953125, 28.055076599121094, 26.970630645751953, 26.084365844726562, 21.81629180908203, 20.70516586303711, 20.311023712158203, 20.049480438232422, 21.526004791259766, 23.021743774414062, 22.79100799560547, 22.708431243896484, 22.608386993408203, 22.62966537475586, 23.756195068359375, 23.944690704345703, 24.971176147460938, 25.603836059570312, 25.13092803955078, 27.76097869873047, 27.128318786621094, 25.847911834716797, 25.847911834716797, 26.939823150634766, 27.689678192138672, 26.63619613647461, 25.730716705322266, 24.498268127441406, 22.460384368896484, 22.59298324584961, 22.421955108642578, 22.529621124267578, 22.504531860351562, 22.545024871826172, 22.781478881835938, 22.80069351196289, 23.292816162109375, 22.766075134277344, 23.652339935302734, 23.125598907470703, 22.919952392578125, 23.860050201416016, 23.923412322998047, 23.721576690673828, 22.529621124267578]
    target= [31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328]
    date= ['20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401']


    # predict= [17.353925704956055, 17.74806785583496, 17.19147300720215, 16.466629028320312, 16.28972625732422, 16.19071388244629, 16.132116317749023, 15.97927188873291, 16.087890625, 15.968632698059082, 15.781168937683105, 15.708121299743652, 15.584097862243652, 15.92543888092041, 15.6187162399292, 15.441813468933105, 15.33422565460205, 15.274596214294434, 15.214966773986816, 15.471667289733887, 15.43704891204834, 15.481274604797363, 15.299607276916504, 15.008288383483887, 15.431252479553223, 15.3331937789917, 15.35820484161377, 15.667706489562988, 15.69374942779541, 15.461028099060059, 14.895857810974121, 14.640189170837402, 14.330687522888184, 14.280665397644043, 14.26526165008545, 14.49695110321045, 14.604538917541504, 14.590167045593262, 15.968632698059082, 16.072486877441406, 14.94484806060791, 14.673775672912598, 14.477736473083496, 14.383488655090332, 14.206664085388184, 14.078829765319824, 13.790606498718262, 13.653164863586426, 13.3590669631958, 13.30904483795166, 13.624342918395996, 13.727166175842285, 13.477293968200684, 13.353270530700684, 13.318652153015137, 13.285065650939941, 13.48690128326416, 13.384078025817871, 13.58972454071045, 13.633950233459473, 13.683972358703613, 13.66856861114502, 13.649353981018066, 13.624342918395996, 13.64355754852295, 14.925711631774902, 15.447688102722168, 14.267325401306152, 15.162165641784668, 18.33928108215332, 18.201839447021484, 19.020931243896484, 17.77307891845703, 19.639934539794922, 17.817304611206055, 17.1568546295166, 17.043392181396484, 17.13763999938965, 18.013343811035156, 16.644485473632812, 16.06287956237793, 17.964353561401367, 17.792293548583984, 18.03359031677246, 18.271076202392578, 18.394145965576172, 18.38827133178711, 18.664186477661133, 18.634410858154297, 18.403675079345703, 18.340312957763672, 18.422889709472656, 18.271076202392578, 18.472911834716797, 18.43829345703125, 18.48728370666504, 17.275081634521484, 17.087617874145508, 17.166461944580078, 16.096466064453125, 15.993643760681152, 17.688438415527344, 17.59522247314453, 17.80190086364746, 18.13260269165039, 18.255672454833984, 17.500974655151367, 16.21572494506836, 17.876934051513672, 18.047962188720703, 16.60025978088379, 16.890546798706055, 16.72229766845703, 16.21572494506836, 16.042633056640625, 15.856202125549316, 15.752346992492676, 15.728367805480957, 16.08685874938965, 16.102262496948242, 16.03302574157715, 15.910035133361816, 15.771561622619629, 15.427441596984863, 15.72257137298584, 15.367812156677246, 15.550511360168457, 15.687952995300293, 18.068208694458008, 18.728580474853516, 18.063365936279297, 21.295269012451172, 21.84709930419922, 19.31502914428711, 19.399669647216797, 18.970909118652344, 21.045475006103516, 19.88226318359375, 22.12198257446289, 20.635929107666016, 20.649269104003906, 18.857524871826172, 18.561363220214844, 18.113388061523438, 18.063365936279297, 21.062625885009766, 20.157146453857422, 20.35517120361328, 20.220508575439453, 20.226383209228516, 20.05329132080078, 20.03995132446289, 18.857524871826172, 18.531587600708008, 17.285720825195312, 17.08864974975586, 16.9251651763916, 16.83194923400879, 16.658857345581055, 16.507043838500977, 16.466629028320312, 16.284961700439453, 16.275354385375977, 16.250343322753906, 16.9789981842041, 16.669496536254883, 18.477754592895508, 17.398151397705078, 17.3885440826416, 18.832435607910156, 17.560604095458984, 19.984054565429688, 19.83224105834961, 19.739978790283203, 18.940101623535156, 18.767009735107422, 20.680076599121094, 19.15734100341797, 19.72457504272461, 17.166461944580078, 16.811702728271484, 16.531023025512695, 16.358963012695312, 16.255107879638672, 16.16570281982422, 16.003250122070312, 15.9744291305542, 15.856202125549316, 15.708121299743652, 15.653334617614746, 15.520657539367676, 15.441813468933105, 15.41203784942627, 15.6187162399292, 15.288968086242676, 15.445624351501465, 15.539872169494629, 15.3870267868042, 15.239977836608887, 14.959298133850098, 15.211155891418457, 15.15152645111084, 15.042906761169434, 14.664168357849121, 14.467097282409668, 14.36911678314209, 14.354666709899902, 14.383488655090332, 14.34029483795166, 14.274868965148926, 14.230643272399902, 13.907801628112793, 13.966399192810059, 13.779967308044434, 14.905465126037598, 16.231128692626953, 15.692717552185059, 15.195752143859863, 14.138459205627441, 15.13135814666748, 15.520657539367676, 14.821856498718262, 14.783427238464355, 13.932812690734863, 13.642525672912598, 13.574320793151855, 13.549309730529785, 13.599331855773926, 13.638714790344238, 13.663725852966309, 13.549309730529785, 13.770359992980957, 14.374913215637207, 14.831463813781738, 14.880454063415527, 14.168234825134277, 14.624785423278809, 14.649796485900879, 14.777630805969238, 15.46205997467041, 14.341326713562012, 13.932812690734863, 13.583928108215332, 13.72240161895752, 13.752177238464355, 13.746380805969238, 13.446486473083496, 13.624342918395996, 14.282729148864746, 14.660435676574707, 15.95044994354248, 16.963594436645508, 17.098257064819336, 17.506771087646484, 16.915557861328125, 18.732391357421875, 17.628808975219727, 16.944379806518555, 16.64345359802246, 16.5156192779541, 16.737701416015625, 16.880939483642578, 16.797330856323242, 14.930476188659668, 15.51208209991455, 14.630581855773926, 16.078283309936523, 15.49088191986084, 16.097497940063477, 16.737701416015625, 19.093978881835938, 20.326427459716797, 18.993934631347656, 20.491580963134766, 21.506790161132812, 20.670547485351562, 22.25664520263672, 20.566692352294922, 24.058074951171875, 25.976699829101562, 27.276321411132812, 27.710956573486328, 28.18386459350586, 29.81601333618164, 30.744678497314453, 31.748294830322266, 32.56532287597656, 33.253562927246094, 33.43999481201172, 33.487953186035156, 33.432373046875, 33.245941162109375, 33.23434829711914, 33.09587478637695, 33.04585266113281, 32.87863540649414, 32.634559631347656, 32.6845817565918, 32.326805114746094, 31.93472671508789, 31.626972198486328, 31.61935043334961, 31.284915924072266, 30.979225158691406, 30.496631622314453, 29.846820831298828, 31.390518188476562, 30.615890502929688, 30.415802001953125, 31.973155975341797, 32.426849365234375, 31.915512084960938, 31.944255828857422, 32.03270721435547, 31.903919219970703, 31.37130355834961, 31.00796890258789, 31.698272705078125, 32.51530075073242, 32.44606399536133, 31.57901382446289, 30.979225158691406, 30.613826751708984, 29.70834732055664, 29.569873809814453, 32.77096939086914, 33.145896911621094, 33.04585266113281, 32.44606399536133, 30.90792465209961, 31.440540313720703, 29.846820831298828, 28.693294525146484, 28.585628509521484, 27.964561462402344, 28.005054473876953, 29.166202545166016, 28.595157623291016, 29.066158294677734, 27.82815170288086, 27.305221557617188, 27.305221557617188, 27.5224609375, 27.147533416748047, 28.387603759765625, 29.560344696044922, 32.87101364135742, 32.732540130615234, 33.10746765136719, 33.145896911621094, 32.81892776489258, 32.40763473510742, 32.83814239501953, 32.40763473510742, 31.88470458984375, 32.6153450012207, 32.44606399536133, 32.35761260986328, 32.28837585449219, 30.761985778808594, 28.860511779785156, 27.866580963134766, 26.853435516357422, 26.714962005615234, 26.084365844726562, 27.235984802246094, 26.399742126464844, 30.140918731689453, 30.17172622680664, 27.541675567626953, 27.560890197753906, 26.053558349609375, 26.124858856201172, 25.50379180908203, 24.97323989868164, 26.47850799560547, 26.597766876220703, 25.17888641357422, 25.955577850341797, 24.784744262695312, 26.684154510498047, 25.50379180908203, 25.630672454833984, 25.592243194580078, 24.20417022705078, 23.94643783569336, 23.760005950927734, 23.621532440185547, 25.55175018310547, 24.479053497314453, 23.860050201416016, 24.627056121826172, 23.485122680664062, 23.27741241455078, 24.254192352294922, 24.93274688720703, 25.140457153320312, 24.961647033691406, 25.078842163085938, 25.3865966796875, 27.05908203125, 26.853435516357422, 28.72203826904297, 28.977706909179688, 29.452678680419922, 30.53506088256836, 29.875564575195312, 28.92974853515625, 27.728107452392578, 27.04955291748047, 26.44770050048828, 26.27286148071289, 27.011123657226562, 27.649341583251953, 28.585628509521484, 29.096965789794922, 29.885250091552734, 29.058536529541016, 29.639110565185547, 29.23543930053711, 28.654865264892578, 28.604843139648438, 29.018199920654297, 29.254653930664062, 29.569873809814453, 30.771514892578125, 29.727561950683594, 29.18541717529297, 27.966625213623047, 26.999530792236328, 25.425025939941406, 25.661479949951172, 26.872650146484375, 27.45528793334961, 28.891319274902344, 28.616436004638672, 28.733631134033203, 27.816558837890625, 27.336029052734375, 26.270797729492188, 25.778675079345703, 27.591697692871094, 27.491653442382812, 28.477962493896484, 28.427940368652344, 28.074291229248047, 27.966625213623047, 27.305221557617188, 27.36477279663086, 25.217315673828125, 24.734722137451172, 25.847911834716797, 25.367382049560547, 24.992454528808594, 25.61145782470703, 25.45376968383789, 24.753936767578125, 24.371387481689453, 24.390602111816406, 24.084911346435547, 23.844646453857422, 23.485122680664062, 23.021743774414062, 23.071765899658203, 22.781478881835938, 23.021743774414062, 22.869930267333984, 23.725387573242188, 23.79462432861328, 23.642810821533203, 23.588977813720703, 23.863861083984375, 24.35979461669922, 24.20417022705078, 24.901939392089844, 25.64226531982422, 24.892410278320312, 24.646270751953125, 24.371387481689453, 24.431095123291016, 26.10564422607422, 25.828697204589844, 25.680694580078125, 24.904003143310547, 24.50986099243164, 25.059627532958984, 24.884788513183594, 24.686763763427734, 27.88579559326172, 27.305221557617188, 27.355243682861328, 25.523006439208984, 24.034889221191406, 25.128864288330078, 25.04041290283203, 24.87319564819336, 25.947956085205078, 26.440078735351562, 26.055622100830078, 24.1041259765625, 23.69076919555664, 24.421409606933594, 26.29001235961914, 26.142009735107422, 29.89684295654297, 28.82208251953125, 28.055076599121094, 26.970630645751953, 26.084365844726562, 21.81629180908203, 20.70516586303711, 20.311023712158203, 20.049480438232422, 21.526004791259766, 23.021743774414062, 22.79100799560547, 22.708431243896484, 22.608386993408203, 22.62966537475586, 23.756195068359375, 23.944690704345703, 24.971176147460938, 25.603836059570312, 25.13092803955078, 27.76097869873047, 27.128318786621094, 25.847911834716797, 25.847911834716797, 26.939823150634766, 27.689678192138672, 26.63619613647461, 25.730716705322266, 24.498268127441406, 22.460384368896484, 22.59298324584961, 22.421955108642578, 22.529621124267578, 22.504531860351562, 22.545024871826172, 22.781478881835938, 22.80069351196289, 23.292816162109375, 22.766075134277344, 23.652339935302734, 23.125598907470703, 22.919952392578125, 23.860050201416016, 23.923412322998047, 23.721576690673828, 22.529621124267578]
    # target= [18.899999618530273, 18.0, 17.25, 17.0, 16.549999237060547, 16.450000762939453, 16.40999984741211, 17.399999618530273, 16.649999618530273, 16.350000381469727, 15.649999618530273, 15.600000381469727, 17.0, 16.350000381469727, 16.329999923706055, 15.600000381469727, 15.899999618530273, 15.149999618530273, 15.609999656677246, 15.850000381469727, 15.65999984741211, 15.600000381469727, 14.869999885559082, 15.369999885559082, 15.619999885559082, 16.200000762939453, 16.75, 16.75, 14.850000381469727, 14.529999732971191, 14.039999961853027, 13.850000381469727, 13.520000457763672, 12.949999809265137, 12.5, 15.399999618530273, 15.149999618530273, 16.799999237060547, 16.729999542236328, 15.920000076293945, 16.299999237060547, 15.550000190734863, 15.300000190734863, 14.949999809265137, 14.800000190734863, 14.899999618530273, 14.869999885559082, 14.350000381469727, 14.399999618530273, 15.100000381469727, 14.350000381469727, 13.75, 12.850000381469727, 12.699999809265137, 12.5, 12.25, 14.5, 14.229999542236328, 14.149999618530273, 14.579999923706055, 14.899999618530273, 14.399999618530273, 14.550000190734863, 14.350000381469727, 15.300000190734863, 15.079999923706055, 14.25, 15.949999809265137, 18.09000015258789, 18.25, 18.049999237060547, 16.149999618530273, 19.549999237060547, 17.8700008392334, 16.979999542236328, 15.579999923706055, 16.020000457763672, 15.899999618530273, 14.949999809265137, 15.0, 17.200000762939453, 16.920000076293945, 17.399999618530273, 17.690000534057617, 18.200000762939453, 18.200000762939453, 18.25, 17.149999618530273, 16.90999984741211, 16.600000381469727, 16.799999237060547, 16.43000030517578, 16.549999237060547, 16.280000686645508, 16.149999618530273, 15.75, 15.300000190734863, 15.050000190734863, 14.899999618530273, 16.149999618530273, 16.549999237060547, 16.34000015258789, 16.75, 16.600000381469727, 16.030000686645508, 15.699999809265137, 15.0, 14.350000381469727, 14.149999618530273, 14.350000381469727, 14.949999809265137, 14.899999618530273, 14.050000190734863, 13.640000343322754, 13.119999885559082, 12.899999618530273, 12.800000190734863, 12.380000114440918, 15.350000381469727, 15.890000343322754, 15.199999809265137, 14.800000190734863, 14.399999618530273, 14.649999618530273, 14.449999809265137, 14.399999618530273, 15.0, 16.25, 17.450000762939453, 17.450000762939453, 23.25, 19.700000762939453, 19.549999237060547, 18.0, 18.479999542236328, 20.450000762939453, 18.200000762939453, 21.200000762939453, 21.0, 18.8799991607666, 16.979999542236328, 17.3799991607666, 15.949999809265137, 17.850000381469727, 19.920000076293945, 19.649999618530273, 19.850000381469727, 19.5, 18.649999618530273, 18.84000015258789, 19.530000686645508, 18.149999618530273, 17.25, 16.459999084472656, 16.3700008392334, 16.360000610351562, 15.550000190734863, 14.970000267028809, 14.460000038146973, 14.550000190734863, 14.699999809265137, 14.149999618530273, 16.25, 17.270000457763672, 16.610000610351562, 17.6299991607666, 17.299999237060547, 17.530000686645508, 17.93000030517578, 17.15999984741211, 18.25, 19.799999237060547, 19.149999618530273, 17.649999618530273, 18.020000457763672, 20.0, 19.450000762939453, 17.450000762939453, 15.890000343322754, 14.34000015258789, 13.75, 13.75, 16.649999618530273, 16.68000030517578, 16.149999618530273, 16.600000381469727, 16.200000762939453, 16.149999618530273, 15.399999618530273, 15.600000381469727, 15.550000190734863, 15.0, 15.350000381469727, 14.649999618530273, 14.569999694824219, 15.050000190734863, 14.850000381469727, 14.649999618530273, 14.300000190734863, 14.25, 14.0, 14.0, 13.699999809265137, 12.960000038146973, 12.729999542236328, 13.100000381469727, 13.34000015258789, 15.359999656677246, 14.800000190734863, 14.149999618530273, 13.949999809265137, 14.050000190734863, 14.130000114440918, 15.300000190734863, 16.049999237060547, 15.380000114440918, 15.050000190734863, 14.40999984741211, 15.550000190734863, 15.350000381469727, 14.680000305175781, 13.300000190734863, 12.65999984741211, 12.25, 12.430000305175781, 12.050000190734863, 14.649999618530273, 14.829999923706055, 14.800000190734863, 14.550000190734863, 14.600000381469727, 15.109999656677246, 15.5, 14.550000190734863, 14.149999618530273, 15.09000015258789, 14.949999809265137, 16.049999237060547, 14.199999809265137, 13.630000114440918, 13.630000114440918, 13.050000190734863, 13.300000190734863, 12.899999618530273, 12.630000114440918, 12.550000190734863, 12.75, 12.550000190734863, 14.960000038146973, 16.049999237060547, 17.700000762939453, 16.850000381469727, 16.549999237060547, 16.200000762939453, 18.299999237060547, 18.0, 16.799999237060547, 15.800000190734863, 15.5, 16.06999969482422, 15.699999809265137, 15.520000457763672, 14.850000381469727, 15.0, 14.399999618530273, 14.800000190734863, 14.5, 16.399999618530273, 17.049999237060547, 19.75, 21.600000381469727, 22.399999618530273, 26.149999618530273, 26.100000381469727, 25.75, 29.299999237060547, 27.700000762939453, 31.399999618530273, 36.16999816894531, 41.79999923706055, 44.20000076293945, 47.099998474121094, 61.65999984741211, 54.45000076293945, 70.5999984741211, 76.8499984741211, 81.94999694824219, 67.75, 62.0, 47.29999923706055, 46.95000076293945, 51.380001068115234, 46.599998474121094, 53.400001525878906, 49.790000915527344, 48.20000076293945, 50.130001068115234, 47.95000076293945, 45.099998474121094, 41.5, 43.16999816894531, 42.349998474121094, 41.099998474121094, 40.70000076293945, 38.25, 42.400001525878906, 34.650001525878906, 34.79999923706055, 38.45000076293945, 41.400001525878906, 39.849998474121094, 40.04999923706055, 36.95000076293945, 33.95000076293945, 33.849998474121094, 31.799999237060547, 35.150001525878906, 37.400001525878906, 35.650001525878906, 33.84000015258789, 34.650001525878906, 31.899999618530273, 29.40999984741211, 28.200000762939453, 33.95000076293945, 34.349998474121094, 32.349998474121094, 32.119998931884766, 28.799999237060547, 30.299999237060547, 28.700000762939453, 30.350000381469727, 30.239999771118164, 29.799999237060547, 29.049999237060547, 30.549999237060547, 28.850000381469727, 29.899999618530273, 28.149999618530273, 27.0, 26.799999237060547, 25.420000076293945, 26.149999618530273, 27.25, 28.350000381469727, 38.95000076293945, 35.25, 32.95000076293945, 32.79999923706055, 32.79999923706055, 33.150001525878906, 35.150001525878906, 31.399999618530273, 31.950000762939453, 34.150001525878906, 32.25, 34.45000076293945, 32.400001525878906, 30.899999618530273, 29.799999237060547, 29.25, 29.149999618530273, 30.559999465942383, 29.360000610351562, 29.850000381469727, 28.8799991607666, 31.709999084472656, 28.850000381469727, 28.489999771118164, 27.93000030517578, 26.34000015258789, 24.770000457763672, 24.600000381469727, 24.850000381469727, 28.549999237060547, 28.780000686645508, 27.450000762939453, 27.450000762939453, 26.75, 26.75, 26.899999618530273, 26.899999618530273, 25.920000076293945, 25.299999237060547, 24.899999618530273, 24.600000381469727, 23.850000381469727, 24.6299991607666, 23.280000686645508, 23.100000381469727, 23.290000915527344, 21.600000381469727, 21.850000381469727, 21.799999237060547, 25.75, 26.0, 25.850000381469727, 25.5, 26.110000610351562, 26.8700008392334, 26.899999618530273, 28.0, 28.25, 28.75, 35.54999923706055, 29.850000381469727, 30.950000762939453, 28.350000381469727, 28.350000381469727, 26.600000381469727, 25.799999237060547, 25.5, 25.399999618530273, 29.899999618530273, 30.100000381469727, 31.149999618530273, 31.299999237060547, 32.79999923706055, 32.0, 31.270000457763672, 30.899999618530273, 30.0, 30.100000381469727, 30.700000762939453, 31.950000762939453, 30.950000762939453, 31.299999237060547, 30.049999237060547, 27.889999389648438, 26.799999237060547, 26.299999237060547, 26.799999237060547, 27.18000030517578, 27.549999237060547, 27.899999618530273, 28.799999237060547, 28.899999618530273, 30.0, 28.75, 28.799999237060547, 31.5, 31.860000610351562, 35.29999923706055, 34.5, 33.90999984741211, 32.95000076293945, 30.700000762939453, 28.75, 28.0, 25.350000381469727, 25.200000762939453, 24.799999237060547, 24.100000381469727, 25.799999237060547, 23.350000381469727, 22.950000762939453, 23.25, 21.950000762939453, 25.149999618530273, 24.459999084472656, 23.75, 23.549999237060547, 22.75, 22.799999237060547, 22.399999618530273, 22.75, 22.649999618530273, 22.899999618530273, 22.709999084472656, 22.75, 21.5, 22.479999542236328, 22.399999618530273, 23.75, 24.299999237060547, 22.850000381469727, 22.700000762939453, 23.899999618530273, 24.0, 26.350000381469727, 25.600000381469727, 24.600000381469727, 23.850000381469727, 23.579999923706055, 24.799999237060547, 23.75, 23.600000381469727, 26.469999313354492, 25.5, 24.399999618530273, 23.149999618530273, 22.799999237060547, 24.149999618530273, 23.399999618530273, 22.75, 23.100000381469727, 24.600000381469727, 22.899999618530273, 22.5, 24.75, 24.850000381469727, 25.850000381469727, 25.700000762939453, 34.04999923706055, 30.549999237060547, 32.849998474121094, 30.450000762939453, 27.450000762939453, 25.40999984741211, 24.459999084472656, 24.200000762939453, 23.799999237060547, 23.75, 23.850000381469727, 22.899999618530273, 21.399999618530273, 21.700000762939453, 21.700000762939453, 25.950000762939453, 24.899999618530273, 25.5, 25.0, 23.850000381469727, 27.899999618530273, 27.100000381469727, 24.450000762939453, 24.600000381469727, 26.649999618530273, 27.649999618530273, 25.100000381469727, 25.200000762939453, 23.899999618530273, 23.510000228881836, 23.049999237060547, 22.049999237060547, 20.5, 19.799999237060547, 20.200000762939453, 24.149999618530273, 23.549999237060547, 21.799999237060547, 22.6299991607666, 22.899999618530273, 22.25, 21.450000762939453, 21.850000381469727, 21.100000381469727, 20.700000762939453, 19.850000381469727, 19.479999542236328]
    # date= ['20190128', '20190129', '20190130', '20190131', '20190201', '20190204', '20190205', '20190206', '20190207', '20190208', '20190211', '20190212', '20190213', '20190214', '20190215', '20190219', '20190220', '20190221', '20190222', '20190225', '20190226', '20190227', '20190228', '20190301', '20190304', '20190305', '20190306', '20190307', '20190308', '20190311', '20190312', '20190313', '20190314', '20190315', '20190318', '20190319', '20190320', '20190321', '20190322', '20190325', '20190326', '20190327', '20190328', '20190329', '20190401', '20190402', '20190403', '20190404', '20190405', '20190408', '20190409', '20190410', '20190411', '20190412', '20190415', '20190416', '20190417', '20190418', '20190422', '20190423', '20190424', '20190425', '20190426', '20190429', '20190430', '20190501', '20190502', '20190503', '20190506', '20190507', '20190508', '20190509', '20190510', '20190513', '20190514', '20190515', '20190516', '20190517', '20190520', '20190521', '20190522', '20190523', '20190524', '20190528', '20190529', '20190530', '20190531', '20190603', '20190604', '20190605', '20190606', '20190607', '20190610', '20190611', '20190612', '20190613', '20190614', '20190617', '20190618', '20190619', '20190620', '20190621', '20190624', '20190625', '20190626', '20190627', '20190628', '20190701', '20190702', '20190703', '20190705', '20190708', '20190709', '20190710', '20190711', '20190712', '20190715', '20190716', '20190717', '20190718', '20190719', '20190722', '20190723', '20190724', '20190725', '20190726', '20190729', '20190730', '20190731', '20190801', '20190802', '20190805', '20190806', '20190807', '20190808', '20190809', '20190812', '20190813', '20190814', '20190815', '20190816', '20190819', '20190820', '20190821', '20190822', '20190823', '20190826', '20190827', '20190828', '20190829', '20190830', '20190903', '20190904', '20190905', '20190906', '20190909', '20190910', '20190911', '20190912', '20190913', '20190916', '20190917', '20190918', '20190919', '20190920', '20190923', '20190924', '20190925', '20190926', '20190927', '20190930', '20191001', '20191002', '20191003', '20191004', '20191007', '20191008', '20191009', '20191010', '20191011', '20191014', '20191015', '20191016', '20191017', '20191018', '20191021', '20191022', '20191023', '20191024', '20191025', '20191028', '20191029', '20191030', '20191031', '20191101', '20191104', '20191105', '20191106', '20191107', '20191108', '20191111', '20191112', '20191113', '20191114', '20191115', '20191118', '20191119', '20191120', '20191121', '20191122', '20191125', '20191126', '20191127', '20191129', '20191202', '20191203', '20191204', '20191205', '20191206', '20191209', '20191210', '20191211', '20191212', '20191213', '20191216', '20191217', '20191218', '20191219', '20191220', '20191223', '20191224', '20191226', '20191227', '20191230', '20191231', '20200102', '20200103', '20200106', '20200107', '20200108', '20200109', '20200110', '20200113', '20200114', '20200115', '20200116', '20200117', '20200121', '20200122', '20200123', '20200124', '20200127', '20200128', '20200129', '20200130', '20200131', '20200203', '20200204', '20200205', '20200206', '20200207', '20200210', '20200211', '20200212', '20200213', '20200214', '20200218', '20200219', '20200220', '20200221', '20200224', '20200225', '20200226', '20200227', '20200228', '20200302', '20200303', '20200304', '20200305', '20200306', '20200309', '20200310', '20200311', '20200312', '20200313', '20200316', '20200317', '20200318', '20200319', '20200320', '20200323', '20200324', '20200325', '20200326', '20200327', '20200330', '20200331', '20200401', '20200402', '20200403', '20200406', '20200407', '20200408', '20200409', '20200413', '20200414', '20200415', '20200416', '20200417', '20200420', '20200421', '20200422', '20200423', '20200424', '20200427', '20200428', '20200429', '20200430', '20200501', '20200504', '20200505', '20200506', '20200507', '20200508', '20200511', '20200512', '20200513', '20200514', '20200515', '20200518', '20200519', '20200520', '20200521', '20200522', '20200526', '20200527', '20200528', '20200529', '20200601', '20200602', '20200603', '20200604', '20200605', '20200608', '20200609', '20200610', '20200611', '20200612', '20200615', '20200616', '20200617', '20200618', '20200619', '20200622', '20200623', '20200624', '20200625', '20200626', '20200629', '20200630', '20200701', '20200702', '20200706', '20200707', '20200708', '20200709', '20200710', '20200713', '20200714', '20200715', '20200716', '20200717', '20200720', '20200721', '20200722', '20200723', '20200724', '20200727', '20200728', '20200729', '20200730', '20200731', '20200803', '20200804', '20200805', '20200806', '20200807', '20200810', '20200811', '20200812', '20200813', '20200814', '20200817', '20200818', '20200819', '20200820', '20200821', '20200824', '20200825', '20200826', '20200827', '20200828', '20200831', '20200901', '20200902', '20200903', '20200904', '20200908', '20200909', '20200910', '20200911', '20200914', '20200915', '20200916', '20200917', '20200918', '20200921', '20200922', '20200923', '20200924', '20200925', '20200928', '20200929', '20200930', '20201001', '20201002', '20201005', '20201006', '20201007', '20201008', '20201009', '20201012', '20201013', '20201014', '20201015', '20201016', '20201019', '20201020', '20201021', '20201022', '20201023', '20201026', '20201027', '20201028', '20201029', '20201030', '20201102', '20201103', '20201104', '20201105', '20201106', '20201109', '20201110', '20201111', '20201112', '20201113', '20201116', '20201117', '20201118', '20201119', '20201120', '20201123', '20201124', '20201125', '20201127', '20201130', '20201201', '20201202', '20201203', '20201204', '20201207', '20201208', '20201209', '20201210', '20201211', '20201214', '20201215', '20201216', '20201217', '20201218', '20201221', '20201222', '20201223', '20201224', '20201228', '20201229', '20201230', '20201231', '20210104', '20210105', '20210106', '20210107', '20210108', '20210111', '20210112', '20210113', '20210114', '20210115', '20210119', '20210120', '20210121', '20210122', '20210125', '20210126', '20210127', '20210128', '20210129', '20210201', '20210202', '20210203', '20210204', '20210205', '20210208', '20210209', '20210210', '20210211', '20210212', '20210216', '20210217', '20210218', '20210219', '20210222', '20210223', '20210224', '20210225', '20210226', '20210301', '20210302', '20210303', '20210304', '20210305', '20210308', '20210309', '20210310', '20210311', '20210312', '20210315', '20210316', '20210317', '20210318', '20210319', '20210322', '20210323', '20210324', '20210325', '20210326', '20210329', '20210330', '20210331', '20210401']



    plot_title =  'may14_test_1d'
    
    plt.figure(figsize=(17,7))
    plt.plot(date, predict, '--', label = 'predict', alpha=0.6)#####
    plt.plot(date, target, 'r.-', label = 'target', alpha=0.6)#####    
    plt.xlabel('dates')
    plt.xticks(np.arange(0, len(date) + 1, 5))
    plt.xticks(rotation=30)
    plt.ylabel('Future level')
    plt.title(plot_title)
    plt.legend()
    path_dir = './'
    img_name = plot_title +'.jpg'
    final = os.path.join(path_dir, img_name)
    plt.savefig(final)

