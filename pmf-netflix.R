rm(list=ls())
setwd("C:/Users/Lenovo/Code/R/r_pmf_paper")
data=read.table("netflix.txt", header = FALSE)
rating=data[,3]
movie=data[,2]
user=data[,1]
data_num=length(rating)
movie_num= max(movie)#电影数量
user_num=max(user)#用户数量

#parametre
epsilon=50 # Learning rate 学习率
lambda  = 0.1#Regularization parameter正则化参数 
momentum=0.7#动量优化参数
epoch=1#初始化epoch
maxepoch=50#总训练次数
num_feat = 5 #Rank 10 decomposition 隐因子数量
err_train1=rep(0,maxepoch)
err_valid1=rep(0,maxepoch)
err_random=rep(0,maxepoch)


train_num=(data_num*9)%/%10;
train_vec=data[1:train_num,];#训练集
probe_vec=data[train_num:data_num,];#测试集

movie_num= max(movie)#电影数量1682
user_num=max(user)#用户数量943
#sum_list=max(data)
mean_rating = mean(train_vec[,3])#平均评级

pairs_tr = length((train_vec[,3]))#training data 训练集长度
pairs_pr = length((probe_vec[,3]))#validation data 验证集长度

numbatches= 9 #Number of batches把数据分为9份
num_m = movie_num  # Number of movies 电影数量
num_p = user_num  #Number of users 用户数量

#初始化
w1_M1     = 0.1*matrix(runif(num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T) # Movie feature vectors 生成用户物品特征矩阵d=10
w1_P1     = 0.1*matrix(runif(num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T) # User feature vecators
w1_M1_inc = matrix(rep(0,num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T)#生成同shape的全零矩阵
w1_P1_inc = matrix(rep(0,num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T)


for(epoch in epoch:maxepoch){
  #采用mini batch的方法，每次训练9个样本
  for(batch in 1:numbatches){
    N=10000 #number training triplets per batch 每次训练三元组的数量
    aa_p= train_vec[((batch-1)*N+1):(batch*N),1]#读取用户列每次读取一万个
    aa_m= train_vec[((batch-1)*N+1):(batch*N),2]#读取电影列
    rating = train_vec[((batch-1)*N+1):(batch*N),3]#读取评级列
    rating = rating-mean_rating; #Default prediction is the mean rating. 
    # Compute Predictions %%%%%%%%%%%%%%%%%
    pred_out= apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum)#每一行进行求和,.*为对应元素相乘 10k*1，用隐特征矩阵相乘得到1w个预测评级
    f=sum((pred_out - rating)^2 + 0.5*lambda*(apply((w1_M1[aa_m,]^2 + w1_P1[aa_p,]^2),1,sum) ))  
    #求出损失函数
    #Compute Gradients %%%%%%%%%%%%%%%%%%%迭代 
    IO =2*(pred_out - rating)
    kkkk=num_feat-1
    for(kkk in 1:kkkk){
      IO =cbind(IO,2*(pred_out - rating))
    }
    
    #IO = repmat(2*(pred_out - rating),1,num_feat)
    #将损失矩阵的二倍，复制10列
    Ix_m=IO*w1_P1[aa_p,] + lambda*w1_M1[aa_m,]#损失*U-lambda*V 就是更新规则
    Ix_p=IO*w1_M1[aa_m,] + lambda*w1_P1[aa_p,]#损失*V-lambda*U
    dw1_M1 = matrix(rep(0,num_m*num_feat),nrow = num_m , ncol = num_feat ,byrow =T)
    dw1_P1 = matrix(rep(0,num_p*num_feat),nrow = num_p , ncol = num_feat ,byrow =T)#生成全零用户特征矩阵
    
    for(ii in 1:N){#迭代一万次 每一行一行来 得到更新矩阵
      dw1_M1[aa_m[ii],]=  dw1_M1[aa_m[ii],] +  Ix_m[ii,]
      dw1_P1[aa_p[ii],]=  dw1_P1[aa_p[ii],] +  Ix_p[ii,]
    }
    
    # Update movie and user features %%%%%%%%%%%
    w1_M1_inc = momentum*w1_M1_inc + epsilon*dw1_M1/N
    w1_M1 =  w1_M1 - w1_M1_inc#原矩阵-负导数*学习率
    w1_P1_inc = momentum*w1_P1_inc + epsilon*dw1_P1/N
    w1_P1 =  w1_P1 - w1_P1_inc
  }
  #此时所有(9轮)batch结束 
  #现在已经得到了此轮epoch后的一组U和V
  # Compute Predictions after Paramete Updates %%%%%%%%%%%%%%%%%
  pred_out= apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum)
  f_s=sum((pred_out - rating)^2 + 0.5*lambda*(apply((w1_M1[aa_m,]^2 + w1_P1[aa_p,]^2),1,sum) ))  
  err_train1[epoch] = sqrt(f_s/N)
  #Compute predictions on the validation set %%%%%%%%%%%%%%%%%%%%%% 
  NN=pairs_pr#验证集长度
  aa_p = probe_vec[,1]#读取验证集的user、movie、rating
  aa_m = probe_vec[,2]
  rating = probe_vec[,3]
  
  pred_out =apply(w1_M1[aa_m,]*w1_P1[aa_p,],1,sum) + mean_rating#预测结果加上mean才行
  pred_out[pred_out>5]=5
  pred_out[pred_out<1]=1 #使得预测结果超过评分区间的值依旧掉在区间内
  
  err_valid1[epoch]= sqrt(sum((pred_out- rating)^2)/NN)
  RMSE=sqrt(sum((pred_out- rating)^2)/NN)
  MAE=sum(abs(pred_out- rating))/NN
  print(paste('epoch',epoch,'Train RMSE',signif(err_train1[epoch], 4),'Test RMSE',signif(err_valid1[epoch], 4)))
}
print(paste('RMSE',signif(RMSE, 4),'MAE',signif(MAE, 4)))

#计算Pre@5 Re@5
# TOP-N 推荐

j=10
txt=numeric(user_num)
for(i in 1:user_num){
  user_i_rating_real=probe_vec[(probe_vec$V1==i),]
  user_i_rating_real=user_i_rating_real[order(user_i_rating_real$V3,decreasing = TRUE),]
  user_i_rating=w1_P1[i,]%*%t(w1_M1[user_i_rating_real$V2,])+mean_rating
  if(length(user_i_rating_real$V2)>j){
    txt[i]=sum(rank(-1*user_i_rating)[1:j]<=j)/j
  }
  if(length(user_i_rating_real$V2)<=j){
    ti=sum(user_i_rating_real$V3>=4)
    if(ti!=0){
      txt[i]=sum(user_i_rating[1:ti]>=4)/ti
    }
  }
}
Pre=mean(txt)

txt=numeric(user_num)
for(i in 1:user_num){
  user_i_rating_real=probe_vec[(probe_vec$V1==i),]
  user_i_rating_real=user_i_rating_real[order(user_i_rating_real$V3,decreasing = TRUE),]
  user_i_rating=w1_P1[i,]%*%t(w1_M1[user_i_rating_real$V2,])+mean_rating
  if(length(user_i_rating_real$V2)>j){
    user_i_rating[user_i_rating<4]=0
    user_i_rating[user_i_rating>4]=1
    ti=sum(user_i_rating)
    if(ti!=0){
      bigerthan4=sum(user_i_rating_real$V3>=4)
      tinri=sum(user_i_rating[1:bigerthan4])
      txt[i]=tinri/ti
    }
  }
  if(length(user_i_rating_real$V2)<=j){
    ti=sum(user_i_rating_real$V3>=4)
    if(ti!=0){      
      txt[i]=sum(user_i_rating[1:ti]>=4)/ti
    }
  }
}
Re=mean(txt)


print(paste('Re',signif(Re, 4),'Pre',signif(Pre, 4)))
