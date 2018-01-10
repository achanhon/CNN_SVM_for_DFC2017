# SMUGGLING EXAMPLES

This script present the first experiment on smuggling examples (05/01/2018).

Smuggling examples is a technic which allows to bias training toward a specific weight.
This could allow someone to claim following the rigorous train test paradigm while being in reality doing something close to learn on the test set.

Such falsification is only possible which complete access to test set, and thus, should not be a problem in academia where leading benchmark have high standard evaluation process.
However, I want to warm community that such falsification seems completely possible in industry !

## WARNING

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**CODE IS NOT CROSS PLATEFORM - IT HAS BEEN TESTED ON UBUNTU 16.04 ONLY**

**The code is provided for research only and should NEVER be used for commercial purpose.**

If you do not manadge to reproduce the experiment, feel free to ask me about more detail.
I will try to answer **all** requests depending on my available time.

## REQUIREMENT

vgg model : https://github.com/jcjohnson/pytorch-vgg

CIFAR10 : https://www.cs.toronto.edu/~kriz/cifar.html

liblinear2.1 : https://www.csie.ntu.edu.tw/~cjlin/liblinear/

you should create a build folder with a vgg folder containing vgg model



