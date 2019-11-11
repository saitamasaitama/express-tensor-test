import express from 'express';
import * as fs from 'fs';
import * as path from 'path';


interface routesInterface {
  [key:string]:any
}

const routes:routesInterface={};

fs.readdirSync("./controller/").forEach(
  p=>{
    console.log(`PATH:${p} ${path.extname(p)}`);    
    if(path.extname(p)==".ts"){
      const route:string=path.basename(p,".ts");
      console.log(`ROUTE ${route} ADD FROM ./controller/${p}`);
      routes[route]=require(`./controller/${p}`).default;
      
    }
});
console.log("ROUTE DONE");

export default routes;