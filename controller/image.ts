import * as tf from '@tensorflow/tfjs-node'
import express from 'express';
import * as image from '../classes/image'
import multer from 'multer';


const upload=  multer({ dest: './uploads/' });

const router = express.Router()

// middleware that is specific to this router

/*router.use(
  function timeLog (req, res, next) {
    console.log('Time: ', Date.now())
    next()
  }
)
*/
// define the home page route
/**
 * テスト用。画像をアップロードして変換状態を見る
 */
router.post('/file-up', upload.single('image'), (req, res, next) => {
  const size:[number,number]=[
    55,55
  ];

  const tensor=tf.image.resizeBilinear(
    image.image2tensor(req.file.path),size);

  const $input=tf.input({shape:tensor.shape});
  const gray= tf.layers.dense({
    inputShape:tensor.shape,
    units:1,
    useBias:true,
    weights:[
      tf.ones([3,1]).div(3),
      tf.ones([1]).div(3),
    ],
    activation:'relu',
    batchSize:255,
  });

  const gray2=tf.layers.conv2d({
    inputShape:tensor.shape,
    kernelSize:2,
    filters:1,
    weights:[
      tf.ones([2,2,3,1]).div(12)
    ],
    batchSize:255,
    useBias:false,
    activation:'relu',
    padding:'same'
  });
  const avg=tf.layers.averagePooling2d({
    inputShape:tensor.shape,
    poolSize:[3,3],
    strides:1,
  });

  const max=tf.layers.maxPooling2d({
    inputShape:tensor.shape,
    poolSize:[2,2],
    strides:1,
  });

  console.dir(gray2.apply(tf.stack([tensor])));

  return res.render("include/2dA",{
    req:req.file,
    images:[
      tensor,
      gray.apply(tensor),
      gray2.apply(tf.stack([tensor])),
      avg.apply(tf.stack([tensor])),
      max.apply(tf.stack([tensor])),
    ],
    tensor:tensor
  });
});

router.get("gan",(req,res,next)=>{

});
export default router;
