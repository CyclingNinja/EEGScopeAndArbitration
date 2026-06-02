import time
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
import torch
from braindecode.datasets import TUHAbnormal,TUH,BaseConcatDataset
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet, Deep4Net,EEGNetv4,EEGNetv1,EEGResNet,TCN,SleepStagerBlanco2020,USleep,\
                                TIDNet,get_output_shape,HybridNet, SleepStagerChambon2018
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.datautil import load_concat_dataset
from eeg_win_stack.models import ModelFactory
from eeg_win_stack.io.eeg_loading import custom_crop
from eeg_win_stack.io.labeling import relabel
from eeg_win_stack.tools.filters import (
    remove_tuab_from_dataset,
    select_by_duration,
    select_by_channel,
    select_labeled,
)
from eeg_win_stack.io.dataset_builder import DatasetBuilder
from eeg_win_stack.tools.metrics import MCC, con_mat, find_all_zero, weight_function
from eeg_win_stack.tools.paths import findall
from eeg_win_stack.tools.splits import split_data
from eeg_win_stack.config import load

from torch.nn.functional import elu,relu,gelu

import warnings
warnings.filterwarnings("once")

pd.set_option('display.max_columns', 10)

cfg = load()

mne_log_level        = cfg["run"]["mne_log_level"]
n_repetitions        = cfg["run"]["n_repetitions"]
random_state         = cfg["run"]["random_state"]
n_jobs               = cfg["run"]["n_jobs"]

tuab                 = cfg["data"]["tuab"]
tueg                 = cfg["data"]["tueg"]
n_tuab               = cfg["data"]["n_tuab"]
n_tueg               = cfg["data"]["n_tueg"]
n_load               = cfg["data"]["n_load"]
preload              = cfg["data"]["preload"]
tuab_path            = cfg["data"]["tuab_path"]
tueg_path            = cfg["data"]["tueg_path"]
saved_data           = cfg["data"]["save_recordings"]
saved_path           = cfg["data"]["save_recordings_path"]
saved_windows_data   = cfg["data"]["save_windows"]
saved_windows_path   = cfg["data"]["save_windows_path"]
load_saved_data      = cfg["data"]["load_saved_recordings"]
load_saved_windows   = cfg["data"]["load_saved_windows"]
relabel_dataset      = cfg["data"]["relabel_datasets"]
relabel_label        = cfg["data"]["relabel_labels"]

bandpass_filter      = cfg["preprocessing"]["bandpass_filter"]
low_cut_hz           = cfg["preprocessing"]["low_cut_hz"]
high_cut_hz          = cfg["preprocessing"]["high_cut_hz"]
standardization      = cfg["preprocessing"]["standardization"]
factor_new           = cfg["preprocessing"]["factor_new"]
init_block_size      = cfg["preprocessing"]["init_block_size"]
max_abs_val          = cfg["preprocessing"]["max_abs_val"]
sampling_freq        = cfg["preprocessing"]["sampling_freq"]

window_len_s         = cfg["windowing"]["window_len_s"]
multiple             = cfg["windowing"]["multiple"]
tmin                 = cfg["windowing"]["tmin"]
tmax                 = cfg["windowing"].get("tmax")
sec_to_cut           = cfg["windowing"]["sec_to_cut"]
duration_recording_sec = cfg["windowing"]["duration_recording_sec"]
window_stride_samples  = cfg["windowing"].get("window_stride_samples")
channels             = cfg["windowing"]["channels"]

split_way            = cfg["split"]["split_way"]
train_size           = cfg["split"]["train_size"]
valid_size           = cfg["split"]["valid_size"]
test_size            = cfg["split"]["test_size"]
shuffle              = cfg["split"]["shuffle"]

lr                   = cfg["training"]["learning_rate"]
weight_decay         = cfg["training"]["weight_decay"]
batch_size           = cfg["training"]["batch_size"]
n_epochs             = cfg["training"]["n_epochs"]
n_classes            = cfg["training"]["n_classes"]
test_on_eval         = cfg["training"]["test_on_eval"]
earlystopping        = cfg["training"]["earlystopping"]
es_threshold         = cfg["training"]["es_threshold"]
checkpoint_dir       = cfg["training"]["checkpoint_dir"]

log_path             = cfg["output"]["log_path"]
saved_models_path    = cfg["output"]["saved_models_path"]
plot_result          = cfg["output"]["plot_result"]
train_whole_dataset_again = cfg["output"]["train_whole_dataset_again"]
load_pretrained_model = cfg["output"]["load_pretrained_model"]

model_name           = cfg["model"]["name"]
final_conv_length    = cfg["model"]["final_conv_length"]
dropout              = cfg["model"]["dropout"]
activation           = cfg["model"]["deep4"]["activation"]

deep4_n_filters_time    = cfg["model"]["deep4"]["n_filters_time"]
deep4_n_filters_spat    = cfg["model"]["deep4"]["n_filters_spat"]
deep4_filter_time_length = cfg["model"]["deep4"]["filter_time_length"]
deep4_pool_time_length  = cfg["model"]["deep4"]["pool_time_length"]
deep4_pool_time_stride  = cfg["model"]["deep4"]["pool_time_stride"]
deep4_n_filters_2       = cfg["model"]["deep4"]["n_filters_2"]
deep4_filter_length_2   = cfg["model"]["deep4"]["filter_length_2"]
deep4_n_filters_3       = cfg["model"]["deep4"]["n_filters_3"]
deep4_filter_length_3   = cfg["model"]["deep4"]["filter_length_3"]
deep4_n_filters_4       = cfg["model"]["deep4"]["n_filters_4"]
deep4_filter_length_4   = cfg["model"]["deep4"]["filter_length_4"]
deep4_first_pool_mode   = cfg["model"]["deep4"]["first_pool_mode"]
deep4_later_pool_mode   = cfg["model"]["deep4"]["later_pool_mode"]

shallow_n_filters_time  = cfg["model"]["shallow"]["n_filters_time"]
shallow_filter_time_length = cfg["model"]["shallow"]["filter_time_length"]
shallow_n_filters_spat  = cfg["model"]["shallow"]["n_filters_spat"]
shallow_pool_time_length = cfg["model"]["shallow"]["pool_time_length"]
shallow_pool_time_stride = cfg["model"]["shallow"]["pool_time_stride"]
shallow_split_first_layer = cfg["model"]["shallow"]["split_first_layer"]
shallow_batch_norm      = cfg["model"]["shallow"]["batch_norm"]
shallow_batch_norm_alpha = cfg["model"]["shallow"]["batch_norm_alpha"]

tcn_kernel_size      = cfg["model"]["tcn"]["kernel_size"]
tcn_n_blocks         = cfg["model"]["tcn"]["n_blocks"]
tcn_n_filters        = cfg["model"]["tcn"]["n_filters"]
tcn_add_log_softmax  = cfg["model"]["tcn"]["add_log_softmax"]
tcn_last_layer_type  = cfg["model"]["tcn"]["last_layer_type"]

vit_patch_size       = cfg["model"]["vit"]["patch_size"]
vit_dim              = cfg["model"]["vit"]["dim"]
vit_depth            = cfg["model"]["vit"]["depth"]
vit_heads            = cfg["model"]["vit"]["heads"]
vit_mlp_dim          = cfg["model"]["vit"]["mlp_dim"]
vit_emb_dropout      = cfg["model"]["vit"]["emb_dropout"]

remove_attribute = None


with open(log_path,'a') as f:
    writer=csv.writer(f, delimiter=',',lineterminator='\n',)
    writer.writerow([time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))])
    writer.writerow(['train_loss', 'valid_loss',  'train_accuracy',  'valid_accuracy','etl_time','model_training_time',\
     'test_acc','test_precision','test_recall',\
     'n_repetition','random_state','tuab','tueg','n_tuab','n_tueg','n_load','preload','window_len_s',\
     'tuab_path','tueg_path','saved_data','saved_path','saved_windows_data','saved_windows_path',\
     'load_saved_data','load_saved_windows','bandpass_filter','low_cut_hz','high_cut_hz',\
     'standardization','factor_new','init_block_size','n_jobs','n_classes','lr','weight_decay',\
     'batch_size','n_epochs','tmin','tmax','multiple','sec_to_cut','duration_recording_sec','max_abs_val',\
     'sampling_freq','test_on_eval','split_way','train_size','valid_size','test_size','shuffle',\
     'model_name','final_conv_length','window_stride_samples','relabel_dataset','relabel_label',\
     'channels','dropout','precision_per_recording','recall_per_recording',\

     'acc_per_recording','mcc','mcc_per_recording','activation','remove_attribute'])


print(random_state, tuab, tueg, n_tuab, n_tueg, n_load, preload, window_len_s, \
tuab_path, tueg_path, saved_data, saved_path, saved_windows_data, saved_windows_path, \
load_saved_data, load_saved_windows, bandpass_filter, low_cut_hz, high_cut_hz, \
standardization, factor_new, init_block_size, n_jobs, \
tmin, tmax, multiple, sec_to_cut, duration_recording_sec, max_abs_val, \
sampling_freq, test_on_eval, split_way, train_size, valid_size, test_size, shuffle, \
relabel_dataset, relabel_label, \
channels, remove_attribute)

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
print('device:',device)
if cuda:
    torch.backends.cudnn.benchmark = True
torch.set_num_threads(n_jobs)  # Sets the available number of threads

mne.set_log_level(mne_log_level)

data_loading_start = time.time()


windows_ds = DatasetBuilder(
    load_saved_windows=load_saved_windows,
    saved_windows_path=saved_windows_path,
    save_windows=saved_windows_data,
    load_saved_data=load_saved_data,
    saved_data_path=saved_path,
    save_preprocessed=saved_data,
    use_tuab=tuab,
    use_tueg=tueg,
    tuab_path=tuab_path,
    tueg_path=tueg_path,
    n_tuab=n_tuab,
    n_tueg=n_tueg,
    tmin=tmin,
    tmax=tmax,
    channels=channels,
    relabel_label=relabel_label,
    relabel_dataset=relabel_dataset,
    sampling_freq=sampling_freq,
    sec_to_cut=sec_to_cut,
    duration_recording_sec=duration_recording_sec,
    max_abs_val=max_abs_val,
    multiple=multiple,
    bandpass_filter=bandpass_filter,
    low_cut_hz=low_cut_hz,
    high_cut_hz=high_cut_hz,
    standardization=standardization,
    factor_new=factor_new,
    init_block_size=init_block_size,
    window_len_s=window_len_s,
    window_stride_samples=window_stride_samples,
    n_load=n_load,
    preload=preload,
    n_jobs=n_jobs,
).build()
window_len_samples = windows_ds[0][0].shape[1]

# Split the data:
train_set, valid_set, test_set = split_data(windows_ds, split_way, train_size, valid_size, test_size, shuffle, random_state, remove_attribute)
print('len_valid_train',len(train_set.description.loc[:, ['path']])+len(valid_set.description.loc[:, ['path']]))
print('len_test',len(test_set.description.loc[:, ['path']]))
etl_time = time.time() - data_loading_start

n_channels = windows_ds[0][0].shape[0]

print("n_channels:",n_channels)


for i in range(n_repetitions):
    print(i, random_state, tuab, tueg, n_tuab, n_tueg, n_load, preload, window_len_s, \
          tuab_path, tueg_path, saved_data, saved_path, saved_windows_data, saved_windows_path, \
          load_saved_data, load_saved_windows, bandpass_filter, low_cut_hz, high_cut_hz, \
          standardization, factor_new, init_block_size, n_jobs, n_classes, lr, weight_decay, \
          batch_size, n_epochs, tmin, tmax, multiple, sec_to_cut, duration_recording_sec, max_abs_val, \
          sampling_freq, test_on_eval, split_way, train_size, valid_size, test_size, shuffle, \
          model_name, final_conv_length, window_stride_samples, relabel_dataset, relabel_label, \
          channels)
    if shuffle and i>0:
        # Re-split the data to ensure each repetition uses a different split:
        train_set, valid_set, test_set = split_data(windows_ds, split_way, train_size, valid_size, test_size,
                                                    shuffle, random_state+i)


    mne.set_log_level(mne_log_level)
    def exp(dropout=0.2):
        if activation=='elu':  #choose the activation function
            nonlin=elu
        elif activation=='relu':
            nonlin=relu
        elif activation=='gelu':
            nonlin=gelu

        #select the model(first-stage)
        if model_name in ModelFactory.available():
            model = ModelFactory.create(
                model_name,
                n_channels=n_channels,
                n_classes=n_classes,
                input_window_samples=window_len_samples,
                drop_prob=dropout,
                final_conv_length=final_conv_length,
                # deep4
                deep4_n_filters_time=deep4_n_filters_time,
                deep4_n_filters_spat=deep4_n_filters_spat,
                deep4_filter_time_length=deep4_filter_time_length,
                deep4_pool_time_length=deep4_pool_time_length,
                deep4_pool_time_stride=deep4_pool_time_stride,
                deep4_n_filters_2=deep4_n_filters_2,
                deep4_filter_length_2=deep4_filter_length_2,
                deep4_n_filters_3=deep4_n_filters_3,
                deep4_filter_length_3=deep4_filter_length_3,
                deep4_n_filters_4=deep4_n_filters_4,
                deep4_filter_length_4=deep4_filter_length_4,
                deep4_first_pool_mode=deep4_first_pool_mode,
                deep4_later_pool_mode=deep4_later_pool_mode,
                first_nonlin=nonlin,
                later_nonlin=nonlin,
                # shallow_smac
                shallow_n_filters_time=shallow_n_filters_time,
                shallow_filter_time_length=shallow_filter_time_length,
                shallow_n_filters_spat=shallow_n_filters_spat,
                shallow_pool_time_length=shallow_pool_time_length,
                shallow_pool_time_stride=shallow_pool_time_stride,
                shallow_split_first_layer=shallow_split_first_layer,
                shallow_batch_norm=shallow_batch_norm,
                shallow_batch_norm_alpha=shallow_batch_norm_alpha,
                # tcn_1
                n_blocks=tcn_n_blocks,
                n_filters=tcn_n_filters,
                kernel_size=tcn_kernel_size,
                add_log_softmax=tcn_add_log_softmax,
                last_layer_type=tcn_last_layer_type,
                # vit
                patch_size=vit_patch_size,
                dim=vit_dim,
                depth=vit_depth,
                heads=vit_heads,
                mlp_dim=vit_mlp_dim,
                emb_dropout=vit_emb_dropout,
                # sleep2020 / sleep2018 / usleep
                sampling_freq=sampling_freq,
            )




        if cuda:
            model.cuda()
        training_setup_end = time.time()

        # Start training loop
        model_training_start = time.time()

        from skorch.callbacks import LRScheduler
        from skorch.helper import predefined_split
        from braindecode import EEGClassifier
        from skorch.callbacks import Checkpoint,EarlyStopping

        # set the learning rate scheduler
        monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
        cp = Checkpoint(monitor=monitor,dirname='', f_criterion=None, f_optimizer=None, load_best=False)
        callbacks=["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR' ,T_max=n_epochs - 1)),("cp",cp)] #'CosineAnnealingWarmRestarts', T_0=10 'CosineAnnealingLR' ,T_max=n_epochs - 1
        if earlystopping:
            es_patience=n_epochs//3
            es=EarlyStopping(threshold=es_threshold, threshold_mode='rel', patience=es_patience)
            callbacks.append(('es',es))

        #Set various parameters for training
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss(weight_function(train_set.get_metadata().target,device)),
            optimizer=torch.optim.AdamW,
            train_split=predefined_split(valid_set) if test_on_eval else None,  # using valid_set for validation
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=callbacks,
            device=device,
        )


        # Prevent GPU memory fragmentation
        torch.cuda.empty_cache()

        # Model training for a specified number of epochs. `y` is None as it is already supplied
        # in the dataset.
        global i
        if not load_pretrained_model: # Choose to load a model or train a model
            clf.fit(train_set, y=None, epochs=n_epochs)
            clf.save_params('./saved_models/'+model_name+time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))+'params.pt')

        else:
            clf.initialize()
            clf.load_params('./saved_models/'+params[i])
        model_training_time = time.time() - model_training_start

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        if not load_pretrained_model:
            # Extract loss and accuracy values for plotting from history object
            results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']

            df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                              index=clf.history[:, 'epoch'])
            # get percent of misclass for better visual comparison to loss
            df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                           valid_misclass=100 - 100 * df.valid_accuracy)
            print(df)
            if plot_result: # whether plot the result
                plt.style.use('seaborn')
                fig, ax1 = plt.subplots(figsize=(8, 3))
                df.loc[:, ['train_loss', 'valid_loss']].plot(
                    ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False, fontsize=14)

                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
                ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                df.loc[:, ['train_misclass', 'valid_misclass']].plot(
                    ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
                ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
                ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
                ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
                ax1.set_xlabel("Epoch", fontsize=14)

                # where some data has already been plotted to ax
                handles = []
                handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle='-', label='Train'))
                handles.append(Line2D([0], [0], color='black', linewidth=1, linestyle=':', label='Valid'))
                plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
                plt.tight_layout()
                plt.show()

        from sklearn.metrics import confusion_matrix
        from braindecode.visualization import plot_confusion_matrix

        # test on the testset
        print('test:',test_set.description)
        y_true = test_set.get_metadata().target
        starts=find_all_zero(test_set.get_metadata()['i_window_in_trial'].tolist())
        y_pred = clf.predict(test_set)
        y_pred_proba=clf.predict_proba(test_set)
        print('diff:',sum((np.exp(np.array(y_pred_proba[:,1]))>0.5)!=y_pred))

        # generate confusion matrices
        confusion_mat_per_recording=con_mat(starts,y_true,y_pred)
        confusion_mat_per_recording_proba=con_mat(starts,y_true,y_pred,True,y_pred_proba)
        print(confusion_mat_per_recording)
        print(confusion_mat_per_recording_proba)


        confusion_mat = confusion_matrix(y_true, y_pred)
        print(confusion_mat)


        # generate various evaluation index
        precision=confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[1,0])
        recall=confusion_mat[0,0]/(confusion_mat[0,0]+confusion_mat[0,1])
        acc=(confusion_mat[0,0]+confusion_mat[1,1])/(confusion_mat[0,0]+confusion_mat[0,1]+confusion_mat[1,1]+confusion_mat[1,0])
        mcc=MCC(confusion_mat)
        precision_per_recording=confusion_mat_per_recording[0,0]/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[1,0])
        recall_per_recording=confusion_mat_per_recording[0,0]/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[0,1])
        acc_per_recording=(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[1,1])/(confusion_mat_per_recording[0,0]+confusion_mat_per_recording[0,1]+confusion_mat_per_recording[1,1]+confusion_mat_per_recording[1,0])
        mcc_per_recording=MCC(confusion_mat_per_recording)
        end=time.time()
        print('precision:',precision)
        print('recall:',recall)
        print('acc:',acc)
        print('mcc:',mcc)
        print('precision_per_recording:', precision_per_recording)
        print('recall_per_recording:', recall_per_recording)
        print('acc_per_recording:', acc_per_recording)
        print('mcc:',mcc_per_recording)
        print('etl_time:',etl_time)
        print('model_training_time:',model_training_time)

        with open(log_path, 'a') as f: # save the results
            writer = csv.writer(f, delimiter=',', lineterminator='\n', )
            if not load_pretrained_model:
                his_len=len(df)
                for i2 in range(his_len-1):
                    writer.writerow([df.loc[i2+1][0],df.loc[i2+1][1],df.loc[i2+1][2],df.loc[i2+1][3]])


                writer.writerow([df.loc[his_len][0],df.loc[his_len][1],df.loc[his_len][2],df.loc[his_len][3],etl_time,\
             model_training_time,acc,precision,recall,i,random_state,tuab,tueg,n_tuab,n_tueg,n_load,preload,\
             window_len_s,tuab_path,tueg_path,saved_data,saved_path,saved_windows_data,saved_windows_path,\
             load_saved_data,load_saved_windows,bandpass_filter,low_cut_hz,high_cut_hz,\
             standardization,factor_new,init_block_size,n_jobs,n_classes,lr,weight_decay,\
             batch_size,n_epochs,tmin,tmax,multiple,sec_to_cut,duration_recording_sec,max_abs_val,\
             sampling_freq,test_on_eval,split_way,train_size,valid_size,test_size,shuffle,\
             model_name,final_conv_length,window_stride_samples,relabel_dataset,relabel_label,\
             channels,dropout, precision_per_recording,recall_per_recording,acc_per_recording,mcc,mcc_per_recording,activation,remove_attribute])
            else:
                writer.writerow(['test_model','test_model','test_model','test_model',etl_time,\
             model_training_time,acc,precision,recall,i,random_state,tuab,tueg,n_tuab,n_tueg,n_load,preload,\
             window_len_s,tuab_path,tueg_path,saved_data,saved_path,saved_windows_data,saved_windows_path,\
             load_saved_data,load_saved_windows,bandpass_filter,low_cut_hz,high_cut_hz,\
             standardization,factor_new,init_block_size,n_jobs,n_classes,lr,weight_decay,\
             batch_size,n_epochs,tmin,tmax,multiple,sec_to_cut,duration_recording_sec,max_abs_val,\
             sampling_freq,test_on_eval,split_way,train_size,valid_size,test_size,shuffle,\
             model_name,final_conv_length,window_stride_samples,relabel_dataset,relabel_label,\
             channels,dropout, precision_per_recording,recall_per_recording,acc_per_recording,mcc,mcc_per_recording,activation,remove_attribute])


        if plot_result:
            labels=['normal','abnormal']
            # plot the basic conf. matrix
            plot_confusion_matrix(confusion_mat, class_names=labels) #if there is something wrong, change the version of matplotlib to 3.0.3, or find the result in confusion_mat
            plt.show()
            plot_confusion_matrix(confusion_mat_per_recording, class_names=labels)
            plt.show()

        if train_whole_dataset_again: #Store all the information needed for the second stage model
            with open('./training_detail.csv', 'a') as f1:
                print('len_train_valid',len(train_set)+len(valid_set))
                print('len_test',len(test_set))

                writer1 = csv.writer(f1, delimiter=',', lineterminator='\n', )
                writer1.writerow([time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))])

                windows_true =list(train_set.get_metadata().target)+list(valid_set.get_metadata().target)+list(test_set.get_metadata().target) #save labels
                len_true=len(windows_true)
                for i in range(len_true//16384):
                    writer1.writerow(windows_true[i*16384:(i+1)*16384])
                writer1.writerow(windows_true[(len_true)//16384 * 16384:])

                windows_pred=np.exp(np.concatenate((np.array(clf.predict_proba(train_set)[:,1]),np.array(clf.predict_proba(valid_set)[:,1]),np.array(clf.predict_proba(test_set)[:,1])))) #Store predicted probabilities for all data

                for i in range(len_true//16384):
                    writer1.writerow(windows_pred[i*16384:(i+1)*16384])
                writer1.writerow(windows_pred[(len_true)//16384 * 16384:])


                #Store how many windows each recording has
                len_train=len(list(train_set.get_metadata().target))
                len_valid_train=len(list(valid_set.get_metadata().target))+len_train
                writer1.writerow(find_all_zero(train_set.get_metadata()['i_window_in_trial'].tolist())+[x+len_train for x in find_all_zero(valid_set.get_metadata()['i_window_in_trial'].tolist())]+[y+len_valid_train for y in find_all_zero(test_set.get_metadata()['i_window_in_trial'].tolist())])


                #Store the session and patient to which each recording belongs
                paths = np.array(train_set.description.loc[:, ['path']]).tolist()+np.array(valid_set.description.loc[:, ['path']]).tolist()+np.array(test_set.description.loc[:, ['path']]).tolist()
                patients = []
                sessions = []
                for i in range(len(paths)):
                    splits = paths[i][0].split('\\')
                    patients.append(splits[-3])
                    sessions.append(splits[-2])
                print('patients', patients)
                print('sessions', sessions)
                writer1.writerow(patients)
                writer1.writerow(sessions)


        return acc


    exp(dropout=dropout)
