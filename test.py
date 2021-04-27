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
    predict= [34.61949920654297, 34.62989807128906, 34.31929397583008, 33.02422332763672, 32.800140380859375, 33.68708801269531, 34.146888732910156, 33.78922653198242, 35.137306213378906, 36.06589889526367, 35.88064193725586, 35.958831787109375, 33.27448654174805, 31.49070167541504, 32.49898147583008, 34.26853942871094, 35.16482162475586, 37.849525451660156, 37.8767204284668, 28.502063751220703, 32.46813201904297, 25.56255340576172, 25.60923957824707, 24.135486602783203, 25.29348373413086, 26.38566780090332, 23.396738052368164, 23.989213943481445, 30.44964027404785, 41.73100662231445, 40.36648178100586, 37.30907440185547, 35.32592010498047, 36.60902786254883]
    target= [32.119998931884766, 29.31999969482422, 27.81999969482422, 33.08000183105469, 35.08000183105469, 32.380001068115234, 32.119998931884766, 28.920000076293945, 30.75, 28.110000610351562, 31.020000457763672, 30.3799991607666, 29.6200008392334, 29.079999923706055, 30.31999969482422, 28.81999969482422, 29.520000457763672, 28.219999313354492, 27.0, 26.719999313354492, 25.31999969482422, 26.020000457763672, 27.6200008392334, 27.479999542236328, 40.619998931884766, 35.08000183105469, 33.900001525878906, 33.099998474121094, 33.439998626708984, 33.58000183105469, 35.279998779296875, 31.6200008392334, 31.920000076293945, 33.58000183105469]
    date= ['04/30/2020', '05/01/2020', '05/04/2020', '05/05/2020', '05/06/2020', '05/07/2020', '05/08/2020', '05/11/2020', '05/12/2020', '05/13/2020', '05/14/2020', '05/15/2020', '05/18/2020', '05/19/2020', '05/20/2020', '05/21/2020', '05/22/2020', '05/26/2020', '05/27/2020', '05/28/2020', '05/29/2020', '06/01/2020', '06/02/2020', '06/03/2020', '06/04/2020', '06/05/2020', '06/08/2020', '06/09/2020', '06/10/2020', '06/11/2020', '06/12/2020', '06/15/2020', '06/16/2020', '06/17/2020']
    plot_title =  'test_5d'
    
    plt.figure(figsize=(18,10))
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

