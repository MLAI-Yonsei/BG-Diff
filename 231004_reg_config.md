## erm train
# erm best selection

## val best
# if args.train_fold == 0:
#     best_eta=0.01; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=3; args.t_sampler="loss-second-moment"
# elif args.train_fold == 1:
#     best_eta=0.01; best_lr=1e-05; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=3; args.t_sampler="train-step"
# elif args.train_fold == 2:
#     best_eta=0.01; best_lr=1e-05; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=3; args.t_sampler="uniform"
# elif args.train_fold == 3:
#     best_eta=0.01; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=2; args.t_sampler="uniform"
# elif args.train_fold == 4:
#     best_eta=0.01; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=2; args.t_sampler="uniform"

## gal train val best

if args.train_fold == 0:
    best_eta=0.01; best_lr=1e-05; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=12; args.t_sampler="train-step"; args.wd =0.0001 ;args.nblock=8
elif args.train_fold == 1:
    best_eta=0.01; best_lr=1e-05; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=4; args.t_sampler="train-step"; args.wd =0.0001 ;args.nblock=8
elif args.train_fold == 2:
    best_eta=0.01; best_lr=0.001; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=4; args.t_sampler="train-step"; args.wd =0.001 ;args.nblock=8
elif args.train_fold == 3:
    best_eta=0.01; best_lr=1e-05; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=3; args.t_sampler="train-step"; args.wd =0.001 ;args.nblock=8
elif args.train_fold == 4:
    best_eta=0.01; best_lr=0.0001; args.regressor_epoch=2000; args.diffusion_time_steps=2000; args.final_layers=4; args.t_sampler="uniform"; args.wd =0.0001 ;args.nblock=8


## gal train
## erm best selection
### model param   

#### fold 0
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 12             
g_mlp_layers    | 2              
concat_label_mlp | 1              

#### fold 1
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 1              

#### fold 2
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 1              

#### fold 3
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 12             
g_mlp_layers    | 2              
concat_label_mlp | 0              

#### fold 4
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 1              

## gal best selection 
### model param

#### fold 0
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | uniform        
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 1              

#### fold 1
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 1              

#### fold 2
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 1              

#### fold 3
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 12             
g_mlp_layers    | 2              
concat_label_mlp | 0              

#### fold 4
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 1              

## worst best selection
### model param

#### fold 0
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | front          
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 0              

#### fold 1
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 1              

#### fold 2
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 2              
concat_label_mlp | 1              

#### fold 3
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | front          
t_scheduling    | uniform        
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 0              

#### fold 4
------------------------------
Config Key      | Config Value   
------------------------------
g_pos           | rear           
t_scheduling    | train-step     
final_layers    | 4              
g_mlp_layers    | 4              
concat_label_mlp | 1 