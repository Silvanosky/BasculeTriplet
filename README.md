# BasculeTriplet

## Description

This project propose a method to compute the "Bascule" or seesaw between two
oriented block linked by spanning triplet.

Examples imitate a classical [MicMac](https://github.com/micmacIGN/micmac) chantier with two block and the triplet directory.

It will read all the triplets and select the spanning ones (each side of the
triplet is in the block, and at least one in each block).

## Virtual env
### Create virtual env
```
python -m venv local
```
### Load virtual env
```
source local/bin/activate
pip install -r requirements.txt
```

## Examples

### Run the first example dataset

The *test1* dataset consist of 6 empty image with random orientations.
And 2 perfect triplets, with 2 images of each triplet in the first block and 1
image in the second block.

```console
./main.py ./data/test1/NewOriTmpQuick ./data/test1/Ori-1 ./data/test1/Ori-2
```

You can apply a custom bascule to the second block to test :
```console
./main.py ./data/test1/NewOriTmpQuick ./data/test1/Ori-1 ./data/test1/Ori-2 test
```

You should get this output:
```console
EulerRot [0.00000038 9.99999853 0.00000002]
 Bascule [[ 0.98480776 -0.          0.17364815]
 [ 0.          1.         -0.00000001]
 [-0.17364815  0.00000001  0.98480776]]
 Lambda 0.09999999444968828
 Tr [-9.9999994   0.00000055  0.00000135]
---------------------
Image:  image03.tif :
DiffRot [-0.00000037  0.00000147 -0.00000009]
OriTr [-0.          0.00000009  0.00000016]
---------------------
Image:  image06.tif :
DiffRot [-0.00000037  0.00000147 -0.00000009]
OriTr [ 0.00000033 -0.00000018  0.00000089]
```


