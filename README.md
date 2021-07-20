![SMLFDL](https://socialify.git.ci/amirHossein-Ebrahimi/SMLFDL/image?description=1&descriptionEditable=SVMs%20multi-class%20loss%20feedback%20based%20discriminative%20dictionary%20learning&font=Inter&language=1&pattern=Plus&theme=Light)


## 📄[article][SMLFDL_article]
SVMs multi-class loss feedback based discriminative dictionary learning for image classification

> SMLFDL integrates dictionary learning and support vector machines training into a unified learning
framework by looping the designed multi-class loss term, which
is inspired by the feedback mechanism in cybernetics.

analysis has been done on scene-15 dataset.   
Feature vectors has been prepared by four-level `spatial pyramid`, dense `DAISY` feature description followed by PCA.  
As article proposed SMLFDL are faster in predictions and converge in lower epochs.  
<sub>code for features will be added soon.</sub>


## Highlights:
- Inspired by the feedback mechanism in cybernetics, a novel discriminative dictionary learning framework, named support vector machines (SVMs) multi-class loss feedback based discriminative dictionary learning (SMLFDL) is proposed to learn a dictionary while training SVMs. As far as we know, it is the first time that the feedback mechanism in cybernetics is adopted for constructing dictionary learning model.

- SMLFDL further employ the Fisher discrimination criterion on the coding coefficients under -norm constraint to make the coding coefficients have small intra-class scatter but big inter-class scatter for countering intra-class variability of datasets.

- An efficient and practical SMLFDL optimization algorithm is presented to learn a dictionary while training SVMs. Experimental results on several widely used image databases show that SMLFDL can achieve a competitive performance with other state-of-the-art methods on classification task.

## Notes: 
**The original article was developed in matlab**

The [report file](https://github.com/amirHossein-Ebrahimi/SMLFDL/blob/master/SMLFDL%20report.pdf) is an over-view showing precedures and some figures and didn't published anywhere, it must not be refernece any where, for refernece use [original article][SMLFDL_article]



[SMLFDL_article]: https://www.sciencedirect.com/science/article/abs/pii/S0031320320304933
