This code should help you reimplement the experiments in:

Donahue, J., Hendricks, L. A., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2014). Long-term recurrent convolutional networks for visual recognition and description. arXiv preprint arXiv:1411.4389.
Chicago	

Please see http://www.eecs.berkeley.edu/~lisa_anne/LRCN_video for detailed instructions on how to reimplement experiments and download pre-trained models.



create flow frame using python instead of matlab

https://github.com/pathak22/pyflow
git clone https://github.com/pathak22/pyflow.git
cd pyflow/
python setup.py build_ext -i
python demo.py    # -viz option to visualize output
