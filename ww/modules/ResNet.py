import torch
import torch.nn as nn
# import modules.layers as layers
import layers
# from modules.base import Base
from base import Base
import torch.nn.functional as F

class ResNet20(Base):
    """todo: add L2 regulation manually,dense args"""
    def __init__(self, binW, binA):
        super().__init__(binW, binA)
        self.conv01= layers.BinaryConv2DCL(3,16) if binW else nn.Conv2d(3,16,3,padding=1,bias=False)
        self.conv02= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv03= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv04= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv05= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv06= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv07= layers.BinaryConv2DCL(16,16) if binW else nn.Conv2d(16,16,3,padding=1,bias=False)
        self.conv08= layers.BinaryConv2DCL(16,32) if binW else nn.Conv2d(16,32,3,padding=1,bias=False)
        self.conv09= layers.BinaryConv2DCL(32,32) if binW else nn.Conv2d(32,32,3,padding=1,bias=False)
        self.conv10= layers.BinaryConv2DCL(32,32) if binW else nn.Conv2d(32,32,3,padding=1,bias=False)
        self.conv11= layers.BinaryConv2DCL(32,32) if binW else nn.Conv2d(32,32,3,padding=1,bias=False)
        self.conv12= layers.BinaryConv2DCL(32,32) if binW else nn.Conv2d(32,32,3,padding=1,bias=False)
        self.conv13= layers.BinaryConv2DCL(32,32) if binW else nn.Conv2d(32,32,3,padding=1,bias=False)
        self.conv14= layers.BinaryConv2DCL(32,64) if binW else nn.Conv2d(32,64,3,padding=1,bias=False)
        self.conv15= layers.BinaryConv2DCL(64,64) if binW else nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv16= layers.BinaryConv2DCL(64,64) if binW else nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv17= layers.BinaryConv2DCL(64,64) if binW else nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv18= layers.BinaryConv2DCL(64,64) if binW else nn.Conv2d(64,64,3,padding=1,bias=False)
        self.conv19= layers.BinaryConv2DCL(64,64) if binW else nn.Conv2d(64,64,3,padding=1,bias=False)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.dense=layers.BinaryDense(64,10) if binW else nn.Linear(64,10)
        self.actv01=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv02=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv03=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv04=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv05=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv06=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv07=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv08=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv09=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv10=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv11=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv12=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv13=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv14=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv15=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv16=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv17=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv18=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.actv19=layers.BinaryActivation(False) if binA else nn.ReLU()
        self.batn01=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn02=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn03=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn04=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn05=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn06=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn07=nn.BatchNorm2d(num_features=16, eps=0.001, momentum=0.99)
        self.batn08=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn09=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn10=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn11=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn12=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn13=nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99)
        self.batn14=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
        self.batn15=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
        self.batn16=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
        self.batn17=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
        self.batn18=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
        self.batn19=nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99)
    

    def set_kk(self,kka):
        if not self.binW:
            raise NotImplementedError
        self.conv01.set_kk(kka)
        self.conv02.set_kk(kka)
        self.conv03.set_kk(kka)
        self.conv04.set_kk(kka)
        self.conv05.set_kk(kka)
        self.conv06.set_kk(kka)
        self.conv07.set_kk(kka)
        self.conv08.set_kk(kka)
        self.conv09.set_kk(kka)
        self.conv10.set_kk(kka)
        self.conv11.set_kk(kka)
        self.conv12.set_kk(kka)
        self.conv13.set_kk(kka)
        self.conv14.set_kk(kka)
        self.conv15.set_kk(kka)
        self.conv16.set_kk(kka)
        self.conv17.set_kk(kka)
        self.conv18.set_kk(kka)
        self.conv19.set_kk(kka)
        self.dense.set_kk(kka)

    def set_ka(self, kka):
        if not self.binA:
            raise NotImplementedError
        self.actv01.set_kk(kka)
        self.actv02.set_kk(kka)
        self.actv03.set_kk(kka)
        self.actv04.set_kk(kka)
        self.actv05.set_kk(kka)
        self.actv06.set_kk(kka)
        self.actv07.set_kk(kka)
        self.actv08.set_kk(kka)
        self.actv09.set_kk(kka)
        self.actv10.set_kk(kka)
        self.actv11.set_kk(kka)
        self.actv12.set_kk(kka)
        self.actv13.set_kk(kka)
        self.actv14.set_kk(kka)
        self.actv15.set_kk(kka)
        self.actv16.set_kk(kka)
        self.actv17.set_kk(kka)
        self.actv18.set_kk(kka)
        self.actv19.set_kk(kka)

    def get_kk(self) -> nn.Parameter:
        if not self.binW:
            raise NotImplementedError
        return self.conv02.kk
    
    def get_ka(self) -> nn.Parameter:
        if not self.binW:
            raise NotImplementedError
        return self.actv02.kk

    
    def forward(self,x:torch.Tensor)->torch.Tensor:
            
        x=self.batn01(self.conv01(x))
        x=x+self.conv03(self.actv02(self.batn02(self.conv02(self.actv01(x)))))
        x=x+self.conv05(self.actv04(self.batn04(self.conv04(self.actv03(self.batn03(x))))))
        x=x+self.conv07(self.actv06(self.batn06(self.conv06(self.actv05(self.batn05(x))))))
        x=F.pad(F.avg_pool2d(x,kernel_size=2,stride=2),pad=(int(x.size(3)/4),int(x.size(3)/4),int(x.size(2)/4),int(x.size(2)/4),8,8),mode="constant",value=0)+self.conv09(self.actv08(self.batn08(self.conv08(self.actv07(self.batn07(x))))))
        x=x+self.conv11(self.actv10(self.batn10(self.conv10(self.actv09(self.batn09(x))))))
        x=x+self.conv13(self.actv12(self.batn12(self.conv12(self.actv11(self.batn11(x))))))
        x=F.pad(F.avg_pool2d(x,kernel_size=2,stride=2),pad=(int(x.size(3)/4),int(x.size(3)/4),int(x.size(2)/4),int(x.size(2)/4),16,16),mode="constant",value=0)+self.conv15(self.actv14(self.batn14(self.conv14(self.actv13(self.batn13(x))))))
        x=x+self.conv17(self.actv16(self.batn16(self.conv16(self.actv15(self.batn15(x))))))
        x=x+self.conv19(self.actv18(self.batn18(self.conv18(self.actv17(self.batn17(x))))))
        x=self.actv19(self.batn19(x))
        return self.dense(torch.squeeze(self.avgpool(x)))

if __name__ == '__main__':
    binW=False
    binA=False
    model=ResNet20(binW,binA)
    x=torch.rand((2,3,64,64))
    y=model(x)