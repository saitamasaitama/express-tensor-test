import * as tf from '@tensorflow/tfjs-node'

class Image2Vec extends tf.layers.Layer {
  constructor(args:{}={}) {
    super({});
    this.supportsMasking = true;
  }

  public computeOutputShape(inputShape:number[])
    :number[] {
    console.log("=========ComputeShape!==============");
    return [
      inputShape[0],
      inputShape[1]
    ];
  }

  public call(inputs:tf.Tensor,kwargs:any)
    :tf.Tensor {
    console.log("=========Call!==============");
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
  static get className(){ return 'Image2Vec'; }
}


tf.serialization.registerClass(Image2Vec);  // Needed for serialization.

export default Image2Vec;


