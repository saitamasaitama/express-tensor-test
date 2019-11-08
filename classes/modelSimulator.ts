import * as tf from '@tensorflow/tfjs-node'

class modelSimulator{
  private $model:tf.LayersModel;

  public constructor(model:tf.LayersModel){
    this.$model=model; 
  }



  public Simulate($input?:any):layerSimulateResult[]{
    //結果一覧を作る
    

    const result:layerSimulateResult[]=[
      {
        input:tf.ones([10,10,3]),
        output:tf.ones([10,10,1]),
        current:tf.ones([10,10,1]),
        origin:tf.ones([10,10,3]),
        fromLayer:tf.layers.dense({units:3}),
        toLayer:tf.layers.dense({units:6})
      }
    ];
    return result;
  }

}


interface layerSimulateResult{

  input:tf.Tensor,
  output:tf.Tensor,
  current:tf.Tensor,
  origin:tf.Tensor,
  fromLayer:tf.layers.Layer,
  toLayer:tf.layers.Layer
}

export default modelSimulator;
