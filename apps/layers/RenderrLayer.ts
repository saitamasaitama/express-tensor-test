import * as tf from '@tensorflow/tfjs-node'

interface RenderLayerOpt{
  outputShape:number[]
}


//パスレイヤーを固定のイメージに変換する
class RenderLayer extends tf.layers.Layer {
  constructor() {
    super({});
    this.supportsMasking = true;
  }

  /**
   * 平面図にする
   * @param {*} inputShapes
   */
  public computeOutputShape(inputShape:number[])
    :number[] {
    return [
      inputShape[0],
      inputShape[1],
      inputShape[2],
      2 * inputShape[3]
    ];
  }

  /**
   * Centers the input and applies the following function to every element of
   * the input.
   *
   *     x => [max(x, 0), max(-x, 0)]
   *
   * The theory being that there may be signal in the both negative and positive
   * portions of the input.  Note that this will double the number of channels.
   * @param inputs Tensor to be treated.
   * @param kwargs Only used as a pass through to call hooks.  Unused in this
   *   example code.
   */
  public call(inputs:tf.Tensor,kwargs:any)
    :tf.Tensor {


    return tf.ones([2,3,4]);
  }

  static get className(){ return 'RenderLayer'; }
}


tf.serialization.registerClass(RenderLayer);  // Needed for serialization.

export default RenderLayer;


