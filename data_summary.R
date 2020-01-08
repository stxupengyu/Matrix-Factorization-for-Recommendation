
rm(list=ls())
setwd("C:/Users/Lenovo/Code/R/r_pmf_paper")
data=read.table("epinion.txt", header = FALSE)
rating=data[,3]
movie=data[,2]
user=data[,1]

hist(rating)
data_num=length(rating)
movie_num= max(movie)#电影数量
user_num=max(user)#用户数量

sparse=data_num/(movie_num/10000*user_num)/10000#计算稀疏度

N=data_num/user_num
M=data_num/movie_num

