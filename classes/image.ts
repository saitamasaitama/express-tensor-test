import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";

function image2tensor(
  $file: string,
  $resize: [number, number] = [97, 97]
): tf.Tensor3D {
  try {
    const $image = tf.node.decodeImage(fs.readFileSync($file)) as tf.Tensor3D;
    return $image;
  } catch ($e) {
    console.log(`E:${$file}`);
    return tf.zeros([97, 97, 1]);
  }
}

async function tensor2jpeg($image: tf.Tensor3D, $path: string = "test2.jpg") {
  console.log("BEGIN File Write");
  const bytes: Uint8Array = await tf.node.encodeJpeg($image);
  fs.writeFileSync($path, bytes);
  console.log(`JPG ${$path} write OK!`);
}

export { image2tensor };
