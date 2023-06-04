
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import timm
import albumentations as A
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from thinc.api import to_categorical

################################################################################################################


def unpickle(file):
    
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


batch1 = unpickle('/data')

batch2 = unpickle('/data')

batch3 = unpickle('/data')

batch4 = unpickle('/data')

batch5 = unpickle('/data')

testbatch = unpickle('/data')

images_batch1 = batch1[b"data"]
images_batch2 = batch2[b"data"]
images_batch3 = batch3[b'data']
images_batch4 = batch4[b'data']
images_batch5 = batch5[b'data']

labels_batch1 = batch1[b'labels']
labels_batch2 = batch2[b'labels']
labels_batch3 = batch3[b'labels']
labels_batch4 = batch4[b'labels']
labels_batch5 = batch5[b'labels']

testbatch_images = testbatch[b'data']
testbatch_labels = testbatch[b'labels']

train_images = np.concatenate((images_batch1, images_batch2, images_batch3, images_batch4, images_batch5))
train_labels = np.concatenate((labels_batch1, labels_batch2, labels_batch3, labels_batch4, labels_batch5))

test_images = testbatch_images
test_labels = testbatch_labels

x_train, y_train, x_test, y_test = [], [], [], [] 

for i in range(len(train_images)):
    image = train_images[i].reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)
    x_train.append(image)


for i in range(len(train_labels)):
    y_train.append(train_labels[i])


for i in range(len(test_images)):
    image = test_images[i].reshape(3, 32, 32)
    image = image.transpose(1, 2, 0)
    x_test.append(image)


for i in range(len(test_labels)):
    y_test.append(test_labels[i])

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_train = x_train / 255.0
x_test = x_test / 255.0


def plot_images(images):

    # figure size
    fig = plt.figure(figsize=(10,10))

    # plot image grid
    for x in range(10):
        for y in range(10):
            ax = fig.add_subplot(10, 10, 10*y+x+1)
            plt.imshow(images[10*y+x])
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()
    
    
#plot_images(x_train[:100])

################################################################################################################

########################################DATASET and DATALOADER##################################################

def To_Categorical(label):
    label = label.astype(int)
    label_cat = to_categorical(label, 10)
    return label_cat


class My_Dataset(Dataset):
    def __init__(self, imgs, labls, mode):
        super().__init__()
        
        self.imgs = imgs 
        self.labls = labls
        self.mode = mode
        
        self.train_transforms = A.Compose({
            A.Resize(256, 256),
            #A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.7),  #V1
            #A.RandomResizedCrop(p=0.5, height=256, width=256, scale=(0.9, 1.5), ratio=(0.5, 1.5), interpolation=1),
            

  

        })
        self.test_transforms = A.Compose({
             
            A.Resize(256, 256),

           })

    def __len__(self):
        
        return len(self.imgs)


    def __getitem__(self, index):
        img = self.imgs[index]
        labl = self.labls[index]
        if self.mode == 'train':
      
            data = self.train_transforms(image=img)
            img = data['image']
            img = img.transpose(2, 0, 1)
            
        if self.mode == 'test':
                
                data = self.test_transforms(image=img)
                img = data['image']
                img = img.transpose(2, 0, 1)

        labl = To_Categorical(labl)
        img = torch.Tensor(img)
        labl = torch.Tensor(labl)
        

        return img, labl



train_DS = My_Dataset(x_train, y_train, mode='train')
test_DS = My_Dataset(x_test, y_test, mode='test') 
train_loader = DataLoader(train_DS, shuffle=True, batch_size=128)
test_loader = DataLoader(test_DS, shuffle=True, batch_size=1)



################################################################################################################

                    ################ MODEL #######################
                    
model = timm.create_model('resnet18', pretrained=True, num_classes=10).cuda()

###################################################################################################
                  #####################TRAINING#########################################
                  
        
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,  patience=10, verbose=True)
plot_trainloss = []
plot_trainacc = []
plot_valloss = []
plot_valacc = []    
                                      
def train(epochs):
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        train_correct = 0
        train_total = 0
        val_correct = 0
        val_total = 0

        
        
        print('EPOCH:', epoch + 1, '.....')
        
        for i, (data, labl) in enumerate(train_loader):
            
            model.train()
            data, labl = data.cuda(), labl.cuda()
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, labl)
            train_loss.append(loss.item())
            _, predicted = torch.max(pred.data, 1)
            _, lbl = torch.max(labl.data, 1)
            train_total += labl.size(0)
            train_correct += (predicted == lbl).sum().item()
            train_acc.append((train_correct / train_total) * 100)
            loss.backward()
            optimizer.step()
            
        for i, (data, labl) in enumerate(test_loader):
            with torch.no_grad():
                model.eval()
                
                data, labl = data.cuda(), labl.cuda()
                pred = model(data)
                loss = criterion(pred, labl)
                val_loss.append(loss.item())
                _, predicted = torch.max(pred.data, 1)
                _, lbl = torch.max(labl.data, 1)
                val_total += labl.size(0)
                val_correct += predicted.eq(lbl.data).cpu().sum()
                val_acc.append(( val_correct / val_total) * 100)
                
        
                
                
        mean_train_loss = np.mean(train_loss)
        mean_train_acc = np.mean(train_acc)
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)
        
        #scheduler.step(mean_val_acc)
        
        plot_trainloss.append(mean_train_loss)
        plot_trainacc.append(mean_train_acc)
        plot_valloss.append(mean_val_loss)
        plot_valacc.append(mean_val_acc)
        
        
        
        
        print('TRAIN LOSS:{:.3f} || TRAIN ACCURACY:{:.3f} '.format(mean_train_loss, mean_train_acc) )
        print('VAL LOSS:{:.3f} || VAL ACCUARCY:{:.3f}'.format(mean_val_loss, mean_val_acc) )    
        
        if (best_acc < mean_val_acc):
            best_acc = mean_val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'D:/Course/Deep Learning/model_best.pt')
            print('Model saved at epoch', epoch + 1, 'with Accuracy:{:.2f}'.format(best_acc))
            
        print("Best Accuracy:", best_acc, "Epoch:", best_epoch )
            
            

        
            
   #####################################################################################################          
plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.plot(plot_trainacc, label='Train Acc', color='darkblue', linestyle='dotted')
plt.plot(plot_trainloss, label='Train Loss', color='deeppink', linestyle='dotted')
plt.plot(plot_valacc, label='Test Acc', color='forestgreen', linestyle='dotted')
plt.plot(plot_valloss, label='Test Loss', color='red', linestyle='dotted')
plt.title('Training and validation')
plt.legend()
plt.show()


plt.xlabel("Epochs")
plt.ylabel("Loss and Accuracy")
plt.plot(plot_trainacc[-10:], label='Train Acc', color='darkblue', linestyle='dotted')
plt.plot(plot_trainloss[-10:], label='Train Loss', color='deeppink', linestyle='dotted')
plt.plot(plot_valacc[-10:], label='Test Acc', color='forestgreen', linestyle='dotted')
plt.plot(plot_valloss[-10:], label='Test Loss', color='red', linestyle='dotted')
plt.title('Training and validation')
plt.legend()
plt.show()
###############################################################################################