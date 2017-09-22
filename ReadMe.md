# OFF THE SHELF DEEP LEARNING PIPELINE FOR DFC2017

The provided code allows to replicate some of the experiment from the paper "Off the shelf deep learning pipeline for remote sensing applications" published in BID2017 (http://bigdatafromspace2017.org/).
The code is provided for research only.
(Contact ONERA for commercial purpose)

** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. **

## DATA
The experiment are based on the data fusion context data: https://www.grss-ieee.org/community/technical-committees/data-fusion/2017-ieee-grss-data-fusion-contest-2/

You will need to download the train data to reproduce the experiment

The train data contains osm data (both raster and vector).
You should convert some vectors to raster (see the paper for more detail).
It would lead to produce a map called rgv.png in all osm_raster folder from the training dataset.

However, feel free to ask me for the osm map at "adrien.chan"+'_'+"hon"+'_'+"tong@onera.fr" (it is the 8 dash not the 6 one).
I will try to answer **all** requests.
It will only depend on my available time and the possibility to transfert the 90Mo of the data.

## PIPELINE
The pipeline consists to extract deep feature from OSM data and to concat it with raw image values (means variance for landsat - value for sentinel2).
See the paper for detail.

## COMPILING AND EXECUTING
You need to have cuda compiled + the caffemodel of vgg16 (https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
Adapt the CMakefile to have the path be correct.
L124 of the .cpp -> adapt the path to caffe model of vgg16
In the main function -> adapt the path to the data

## LIMITATION

** CODE IS NOT CROSS PLATEFORM - IT HAS BEEN TESTED ON UBUNTU 16.04 ONLY**

We have observed crucial numerical instability on the OSM data depending on the hardware used to compute the feature **AND** we observed crucial numerical instability between windows/linux.
However, on my computer it leads to the result indicated in the paper.

If you do not manadge to reproduce the experiment, feel free to ask me about more detail.
I will try to answer **all** requests depending on my available time.
