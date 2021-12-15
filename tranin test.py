import os 
import models
import cv2
import torch as t
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import dataset
from config import DefaultConfig
from torchvision import transforms
opt=DefaultConfig()
lr=opt.lr
model=getattr(models,opt.model)
 #transform = transforms.ToTensor() 
datasets=dataset.DogCat(opt.train_data_root,train=True)

#数据的训练

def train(**kwargs):
    #根据命令行参数更新配置
    #step1：模型
    model =getattr(models,opt.model)()    #返回模型类型
    if opt.load_model_path:
        model.load(opt.load_model_path)  #加载模型
#    if opt.load_model_path:
#         model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()       #使用gpu加速模型
    #step2: 数据
#    train_data=dataset.DogCat(opt.train_data_root,train=True)
#    val_data=DogCat(opt.train_data_root,train=False)
    train_dataloaders=DataLoader(datasets,batch_size=2,shuffle=True)   #加载数据

    #step3 目标函数和优化器
    criterion=t.nn.CrossEntropyLoss()   #损失函数
    lr=opt.lr
    optimizer=t.optim.Adam(model.parameters(),lr=0.0005,weight_decay=opt.weight_decay)   #优化器
    s=0
    for eopch in range (20):   #迭代次数 
        s=0
        for ii,(datas,label) in enumerate(train_dataloaders):     
            
            iput=Variable(datas)   #datas是图片数据
            target=Variable(label)  #标签
            if opt.use_gpu:  
                iput=iput.cuda()
                target=target.cuda()
            optimizer.zero_grad()
            score=model(iput)          #  print(score)
            score=t.nn.functional.softmax(score,dim=1).cpu()
            target=target.cpu()
            print(score)
            print(target)

            loss=criterion(score,target)   #计算损失函数
            loss.backward()   #反向传播
            optimizer.step()
          #  print(target)
            _, predicted = t.max(score.data, 1)
            b=predicted.cpu()
            print(b)
            c,b=b.numpy()[0],b.numpy()[1]
            print(c,b)
            loss=loss.cpu()
            s=s+loss.item()
          #print(s)
            
    print("Train Finish")        
    t.save(model.state_dict(), 't.pth')
#train()

#测试数据
def test ():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model=getattr(models,opt.model)()  #打开模型的构造
    model.load_state_dict(t.load('t.pth'))  #加载训练模型的参数
    model.cuda()
    model.eval()
    
    #model.cuda()
    #加载数据
    train_data=dataset.DogCat(opt.test_data_root,test=True)
    test_dataloader=DataLoader(train_data,batch_size=42,shuffle=False,num_workers=0)
    results=[]
    c=[]
    cc=[]
    aa=[]
    with t.no_grad():
        
        for i ,(data,path) in enumerate(test_dataloader):
            
            input=t.autograd.Variable(data)
            input=input.cuda()
            score=model(input)
#            print(score)
#            a=t.nn.functional.softmax(score,dim=1).data.tolist()
            a=t.nn.functional.softmax(score,dim=1).cpu()
            path=path.numpy()
 #           print(a)
            f=a.numpy()
#            print(f)
            for u in range(42):

                
                f=a.numpy()[u][1]
#                print(f)         #得到的是每个bitch里面的是滑坡的概率
                if f>0.9997:
                    aa.append(f)
                    g=path[u]
                    cc.append(g)               #得到的是测试图像的位置（label）
#        print(cc)
#        print(aa)
        ccc=[]                     
        img=cv2.imread(r'E:\yangben\testimg\subset.tif')
        v=img.shape
        H=v[1]
        w=v[0]
        o=[]
        for ii in range(len(cc)):
            o=[]
#            print(ii)
            for z in range(0,w-100,50):               
                for j in range(0,H-100,50):      
                    img_new=img[z:z+100,j:j+100,:]
                    o.append(img_new)
                    k=len(o)
                    if cc[ii]==k:
                        x=z
                        y=j
                        cccc=(x,y)
                        ccc.append(cccc)        #得到的是每个是滑坡区域的左上角的坐标 ，是个列表
        cccc=[]
        ccccc=[]

        for iii in range(len(ccc)):
            for jjj in range(len(ccc)):
                if (ccc[iii][0]==ccc[jjj][0]-50 and ccc[iii][1]==ccc[jjj][1]-50) or  (ccc[iii][0]==ccc[jjj][0]-50 and ccc[iii][1]==ccc[jjj][1]) or  (ccc[iii][0]==ccc[jjj][0] and ccc[iii][1]==ccc[jjj][1]-50) or (ccc[iii][0]==ccc[jjj][0]-50 and ccc[iii][1]==ccc[jjj][1]+50) or (ccc[iii][0]==ccc[jjj][0]+50 and ccc[iii][1]==ccc[jjj][1]-50) or (ccc[iii][0]==ccc[jjj][0]+50 and ccc[iii][1]==ccc[jjj][1])  or (ccc[iii][0]==ccc[jjj][0] and ccc[iii][1]==ccc[jjj][1]+50)   or (ccc[iii][0]==ccc[jjj][0]+50 and ccc[iii][1]==ccc[jjj][1]+50):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                    cccc.append(jjj)
                    ccccc.append(aa[jjj])
#                    print(ccccc)
                    aaa=max(ccccc)
                    for n in range(len(ccccc)):
                    
                        if aa[cccc[n]]==aaa:
                            
                            m=cccc[n]
                            x=int(ccc[m][0])
                            y=int(ccc[m][1])
                            col=100
                            row=100
                            colors=(0,0,255)
                            cv2.rectangle(img,(x,y),(x+row,y+row),colors,5)
                else:
                    x=int(ccc[iii][0])
                    y=int(ccc[iii][1])
                    col=50
                    row=50
                    colors=(0,0,255)
                    cv2.rectangle(img,(x,y),(x+row,y+col),colors,5)
        os.chdir(r'E:\yangben\testimg')
        cv2.imwrite("test1.tif",img)    
        return results
test()







    
    
    


    
    
    

    



    
    
    




    


    

    

    
    

            















                
        
