clc;
clear all;
close all;
data=load('MNISTnumImages5000.txt');
labels=load('MNISTnumLabels5000.txt');
deltaweight1=zeros(784,151);
deltaweight2=zeros(150,785);
momentum=0.7;
batchsize=5;
traindata=cat(2,data(1:4000,:),labels(1:4000,:));
testdata=cat(2,data(4001:5000,:),labels(4001:5000,:));
weight1=normrnd(0,sqrt(2/785),[785,150]);
weight2=normrnd(0,sqrt(2/151),[151,784]);
lr1=0.08;
lr2=0.08;
epc=100;
eachdigitLossAllEpochs=[];
totalLossForeachEpoc=[];
betaq=4;
eachdigitLossAllEpochsNorm=[];
totalLossForeachEpocNorm=[];
lambda=0.0001;


seachdigitLossAllEpochs=[];
stotalLossForeachEpoc=[];

seachdigitLossAllEpochsNorm=[];
stotalLossForeachEpocNorm=[];

indices=randperm(4000);
indices1=randperm(1000);
traindata1=traindata(indices,1:784);
trainlabels=traindata(indices,785); 
testdata1=traindata(indices1,1:784);
testlabels=traindata(indices1,785); 
StotalLossForeachEpocNorm=[];
% 
% weight1=-sqrt(6/(785+150)+(sqrt(6/(785+150))).*rand(785,150));
lengDigits=zeros(1,10);
for b=1:10
   lengDigits(b)=length(trainlabels(trainlabels==b-1));
end

slengDigits=zeros(1,10);

for b1=1:10
   slengDigits(b1)=length(testlabels(testlabels==b1-1));
end
bnu=zeros(1,150);
rowjcap=(bnu)./4000;
row=0.05;

% weight2=-sqrt(6/(151+10)+(sqrt(6/(151+10))).*rand(151,10));
for epoch=1:epc
    
seacherrorAR=zeros(1,10);
conmat=zeros(10,10);
eacherrorAR=zeros(1,10);
for i=1:4000
    if mod(i,batchsize)==0
     ampli=1;
 else
     ampli=0;
 end
traindata1withbias=cat(2,traindata1(i,1:784),1);    
sum1=traindata1withbias*weight1;
activation1=1./(1+exp(-sum1));
rowjcap=rowjcap+activation1;%new%

activation1withbias=cat(2,activation1,1);
sum2=activation1withbias*weight2;
activation2=1./(1+exp(-sum2));
[m mi]=max(activation2);
%
hotcode=zeros(1,10);
sparnesspenality=betaq*(((1-row)./(1-rowjcap)) - (row./rowjcap));

p=trainlabels(i);
hotcode(p+1)=1;
%
% conmat(mi,trainlabels(i)+1)=conmat(mi,trainlabels(i)+1)+1;
% error
error=traindata1(i,1:784)-activation2;
jqw=0.5*sum((error).^2);
deliq=(error.*(activation2.*(1-activation2)))';
deltaweight1=(momentum*deltaweight1)+lr1*deliq*activation1withbias;
weight2=weight2+((deltaweight1').*ampli)-(lambda.*weight2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
deljq=(activation1withbias'.*(1-activation1withbias')).*((weight2*deliq)-betaq*(sparnesspenality));;
deltaweight2=(momentum*deltaweight2)+lr2*deljq(1:150,1)*traindata1withbias;
weight1=weight1+((deltaweight2').*ampli)-(lambda*weight1);
eacherrorAR(trainlabels(i)+1)=eacherrorAR(trainlabels(i)+1)+jqw;
end
rowjcapreal=rowjcap./4000;
eachdigitLossAllEpochs=cat(1,eachdigitLossAllEpochs,eacherrorAR);
totalLossForeachEpoc=cat(2,totalLossForeachEpoc,sum(eacherrorAR));
eachdigitLossAllEpochsNorm=cat(1,eachdigitLossAllEpochsNorm,eacherrorAR./lengDigits);
totalLossForeachEpocNorm=cat(2,totalLossForeachEpocNorm,sum(eacherrorAR)/4000);
% vvv(epoch)=sum(diag(conmat))/4000;
% loss(epoch)=sum(jqw);
for n=1:1000
    testdata1withbias=cat(2,testdata1(n,1:784),1);    
testsum1=testdata1withbias*weight1;
testactivation1=1./(1+exp(-testsum1));
testactivation1withbias=cat(2,testactivation1,1);
testsum2=testactivation1withbias*weight2;
testactivation2=1./(1+exp(-testsum2));
serror=testdata1(n,1:784)-testactivation2;
sjqw=0.5*sum((serror).^2);
seacherrorAR(testlabels(n)+1)=seacherrorAR(testlabels(n)+1)+sjqw;
end
% StotalLossForeachEpocNorm=cat(2,StotalLossForeachEpocNorm,sum(seacherrorAR)/1000);


seachdigitLossAllEpochs=cat(1,seachdigitLossAllEpochs,seacherrorAR);
stotalLossForeachEpoc=cat(2,stotalLossForeachEpoc,sum(seacherrorAR));
seachdigitLossAllEpochsNorm=cat(1,seachdigitLossAllEpochsNorm,seacherrorAR./slengDigits);
stotalLossForeachEpocNorm=cat(2,stotalLossForeachEpocNorm,sum(seacherrorAR)/1000);

end
% loss1=sum(loss);
% 
% for n=1:1000
%     testdata1withbias=cat(2,testdata1(n,1:784),1);    
% testsum1=testdata1withbias*weight1;
% testactivation1=1./(1+exp(-testsum1));
% testactivation1withbias=cat(2,testactivation1,1);
% testsum2=testactivation1withbias*weight2;
% testactivation2=1./(1+exp(-testsum2));
% serror=testdata1(n,1:784)-testactivation2;
% sjqw=0.5*sum((serror).^2);
% seacherrorAR(testlabels(n)+1)=seacherrorAR(testlabels(n)+1)+sjqw;
% end

plot(totalLossForeachEpocNorm);
hold on;
plot(stotalLossForeachEpocNorm);
hold off;

abhi=seachdigitLossAllEpochsNorm(epc,:);
abhi2=eachdigitLossAllEpochsNorm(epc,:);
abhi3=cat(2,abhi2',abhi2');
bar(abhi3);