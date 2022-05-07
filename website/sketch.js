let txt,
  txt2,
  txtCon,
  order = 2,
  len = 80;
ngrams = {};
let img;

function preload() {
  txt = loadStrings("lunyu.txt");
  txt2 = loadStrings("mengzi.txt");
  font = loadFont("chineseWriting.otf");
}

function setup() {
  createCanvas(500, 230);
  background(0);
  textAlign(CENTER);
  
  textFont(font);
  textSize(45);
  fill(245);
  text("論孟",width/2,height/2-5);
  textSize(20);
  text("click to start",width/2,height/2+20);
  
  let txttxt2 = txt + txt2;
  txtCon = RiTa.untokenize(txttxt2);
  txtCon = txtCon.replace(/\s/g, "");
  txtCon = txtCon.replace(/\s/g, "");
  txtCon = txtCon.replace(/\d/g, "");
  txtCon = txtCon.replace(/,,/g, "");
  txtCon = txtCon.replace(/（.）/g, "");
  txtCon = txtCon.replace(/,./g, "");

  for (i = 0; i <= txtCon.length - order; i++) {
    let gram = txtCon.substring(i, i + order);

    if (!ngrams[gram]) {
      ngrams[gram] = [];
    }
    ngrams[gram].push(txtCon.charAt(i + order));
  }
}

function markovIt() {
  let currentGram = txtCon.substring(0, order);
  let result = currentGram;

  for (let i = 0; i < len; i++) {
    let possibilities = ngrams[currentGram];
    let next = random(possibilities);
    result += next;
    currentGram = result.substring(result.length - order, result.length);
  }

  return result;
}

function mouseClicked() {
  background(0);
  let test1 = markovIt();
  for (let i = 0; i < len; i++) {
    if (i % 20 == 0) {
      fill(245);
      text(test1.substring(i, 20 + i), width / 2, 80 + i * 1.2);
    }
  }
  console.log(test1);
}
