import tensorflow as tf
import numpy as np

class Trainer(object):
    def __init__(self, same_wts_ep, mode="b"):
        self.binbest=0.0
        self.same_wts_ep=same_wts_ep
        self.sw_epc=0
        self.tot_sw=0
        self.mode=mode
        self.pmode="a" if mode=="a" else "w"
        self.prto=1.5
        self.maxpush=3
        self.lr_power=0.3
        self.lr_base=1.0
        self.val_bs=125
        self.reduction=.8
        self.break_r=3 #must > 2
        self.bchange=-1 #change mode every time
    
    def refresh(self):
        self.binbest=0.0
        self.sw_epc=0
        self.tot_sw=0

    def train(self, model, datagen, x_tr,y_tr, x_val, y_val, max_epochs, lnr, save_header, clear_optz, x_test, y_test):
        same_push=0
        for epoch_i in range(max_epochs):
            if(self.mode=="w"):
                model.optimizer.learning_rate.assign(lnr/(model.get_kk()/self.lr_base)**self.lr_power)
            else:
                model.optimizer.learning_rate.assign(lnr/(model.get_ka()/self.lr_base)**self.lr_power)

            self.sw_epc += 1
            self.tot_sw +=1
            print("epoch",epoch_i+1, "sw",self.sw_epc, self.tot_sw, "mxep",max_epochs, "lr",model.optimizer.learning_rate.numpy(), "bb",self.binbest)
            if(datagen is not None):
                rst=model.fit(datagen.flow(x_tr,y_tr,batch_size=self.val_bs),
                 epochs=1, validation_data=(x_val,y_val), validation_batch_size=self.val_bs, verbose=2)
            else:
                rst=model.fit(x_tr,y_tr,batch_size=self.val_bs,
                 epochs=1, validation_data=(x_val,y_val), validation_batch_size=self.val_bs, verbose=2)

            #save_weights
            if (self.mode=="b"):
                fmn=save_header+"_w%0.1f_a%0.1f.npz"%(model.get_kk().numpy(),model.get_ka().numpy())
            elif (self.mode=="w"):
                fmn=save_header+"_w%0.1f.npz"%(model.get_kk().numpy())
            else:
                fmn=save_header+"_a%0.1f.npz"%(model.get_ka().numpy())
            wtsn=model.get_weights()
            np.savez(fmn,*wtsn)
            
            vala=rst.history['val_accuracy'][0]

            #test on binary
            if (self.mode=="b" or self.mode=="w"):
                kk_now=tf.identity(model.get_kk().numpy())
                model.set_kk(1e5)
            if (self.mode=="b" or self.mode=="a"):
                ka_now=tf.identity(model.get_ka().numpy())
                model.set_ka(1e5)

            vbin=model.evaluate(x_val,y_val, verbose=0, batch_size=self.val_bs)[1]

            if(self.binbest < vbin):
              self.binbest=vbin
              self.sw_epc = self.tot_sw = 0
              same_push=100 # nextpush will change
              np.savez(save_header + self.mode + "_Best.npz",*wtsn)
              if(x_test is not None):
                  print("\033[94mtest perf: ",end="")
                  model.evaluate(x_test, y_test, verbose=2, batch_size=self.val_bs)
                  print("\033[0m",end="")
            
            if (self.mode=="b" or self.mode=="w"):
                model.set_kk(kk_now)
            if (self.mode=="b" or self.mode=="a"):
                model.set_ka(ka_now)

            if (self.mode=="b" or self.mode=="w"):
                print("kk=",model.get_kk().numpy(),end=" ")
            if (self.mode=="b" or self.mode=="a"):
                print("ka=",model.get_ka().numpy(),end=" ")
            print("val_acc=",vala,"bin_acc=",vbin)

            if(self.tot_sw>=self.same_wts_ep*self.break_r):
              break
            
            #push kk and ka
            if(self.sw_epc>=self.same_wts_ep):
                if(self.mode=="b" and same_push>=self.bchange):
                    self.pmode="a" if self.pmode=="w" else "w"
                    same_push=0
                else:
                    same_push+=1

                if(self.pmode=="w"):
                    for _ in range(self.maxpush):
                        model.set_kk(model.get_kk() * self.prto)
                        vala1=model.evaluate(x_val,y_val,verbose=0,batch_size=self.val_bs)[1]
                        print("push kk to",model.get_kk().numpy(),"acc=",vala1)
                        if(vala1<vala or model.get_kk()>1e3):
                            break
                else:
                    for _ in range(self.maxpush):
                        model.set_ka(model.get_ka() * self.prto)
                        vala1=model.evaluate(x_val,y_val,verbose=0,batch_size=self.val_bs)[1]
                        print("push ka to",model.get_ka().numpy(),"acc=",vala1)
                        if(vala1<vala or model.get_ka()>1e3):
                            break

                self.sw_epc=int(self.sw_epc*self.reduction)

                if(clear_optz):
                    for vari in model.optimizer.variables():
                        vari.assign(tf.zeros_like(vari))

            if (self.mode=="b" or self.mode=="w"):
                if (model.get_kk().numpy()>1e3):
                    break
            if (self.mode=="b" or self.mode=="a"):
                if (model.get_ka().numpy()>1e3):
                    break