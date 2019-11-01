import * as tf from '@tensorflow/tfjs-node'


interface labelCollection{
  [name:string]:any[]
}

class label {
  public constructor(
    name:string,
    data:any
  ){
  }

  
  public ToString():string{
    return "Lemon";   
  }

  public ToTensor():tf.Tensor{
    return tf.tensor([1,2,3]);
  }
}


