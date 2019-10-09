import * as tf from '@tensorflow/tfjs-node'

//整流器？
class Antirectifier extends tf.layers.Layer {
  constructor() {
    super({});
    this.supportsMasking = true;
  }

  /**
   * This layer only works on 4D Tensors 
      [batch, height, width, channels],
   * and produces output with twice as many channels.
   * layer.computeOutputShapes must be overridden in the case that the output
   * shape is not the same as the input shape.
    結果的に入力を倍のサイズにする？
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
    let input = inputs;
    if (Array.isArray(input)) {
      input = input[0];
    }
    this.invokeCallHook(inputs, kwargs);
    const origShape:number[] = input.shape;
    const flatShape =
        [
          origShape[0],
          origShape[1] * origShape[2] * origShape[3]
        ];
    const flattened = input.reshape(flatShape);
    const centered = tf.sub(flattened, flattened.mean(1).expandDims(1));
    const pos = centered.relu().reshape(origShape);
    const neg = centered.neg().relu().reshape(origShape);
    return tf.concat([pos, neg], 3);
  }

  static get className(){ return 'Antirectifier'; }
}


tf.serialization.registerClass(Antirectifier);  // Needed for serialization.

export default Antirectifier;


