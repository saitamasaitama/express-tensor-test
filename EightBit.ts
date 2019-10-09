import * as tf from '@tensorflow/tfjs-node'

//整流器？
class EightBit extends tf.layers.Layer {
  constructor(args: {

  } = {}) {
    super({});
    this.supportsMasking = true;
  }

  /**
  結果を固定のサイズにする
   * @param {*} inputShapes
   */
  public computeOutputShape(inputShape: number[])
    : number[] {
    return [
      256,
      224,
      3,
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
  public call(inputs: tf.Tensor, kwargs: any)
    : tf.Tensor {
    return tf.zeros([256, 224, 3]);
  }

  static get className() { return 'Antirectifier'; }
}


tf.serialization.registerClass(EightBit);  // Needed for serialization.

export default EightBit;


