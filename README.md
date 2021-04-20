<b>1. Problem statement</b>

Writing research code is much different than writing code for the product. In research, one part of code can be written 
and rewritten multiple times with drastic changes, one more significant difference is that, code often written in a 
hurry in notebooks(for convenience and fast experiments) and only part was in file format. 
Because code is rapidly involving and changing it is hard to maintain quality. Not helping the fact that different 
people writing code in different style and quality. 
Forcing one style is hard when there is not much time till a deadline. In the end, you end up with a lot of code and 
branches that should be refactored and rewritten before publishing the code to the public.  
The goal of this project is to make an easily reproducible repository (more precisely continue upgrading existing) 
and good documentation for public use. We have 2 repositories with 15 branches of research code from the ECCV2020 paper 
 Deep vectorization of technical drawings and repository in  Github repository, with more than already 22000 lines of 
code, to combine all the parts. We would like to make the code easy to use and understand and mainly easy to run.   


<b> 2. Main challenges</b>

Combine code from different branches that are not compatible and make an easily readable and reproducible code. Another 
challenge is to make documentation for functions in the code.    

<b>3.Description of a baseline solution. Some other implementations which will serve as inspiration or baseline for your work</b>  
The baseline solution is to release the initial code(code form research) without refactoring(with all branches). 
Another solution is to live the already refactored repository as it is, without any further improvement and commits.  

<b> 3.1. Pros and cons of these solutions</b>

1. Releasing research repositories:   
     Pros: All raw code would be available to anyone with all experiments and legacy code.  
     Cons: Code is a research mess.  Almost nobody would try to understand or try to use it.  
2. Keeping refactored repository as it is:  
     Pros: Time savings.  
     Cons: Not all functionality carefully present and well documented(No good documentation, no docker file, and e.t.c. for more detail look at https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings )  

<b>3.2. Ideas on how to improve it or how you are going to use it.</b>    
     Add Jupyter notebooks with an explanation of how to evaluate functions, add a docker file, and a list of 
     requirements. Add trained models and documentations. For more details look at the list below.
  
<b>4.Roles for the participants</b>  
Because this team consists of only one member, all proposed tasks would be done by me. It includes writing project reports, code, and documentation. In the list below you can find a brief description of tasks:

1. Create code for visualization(rewrite tensorboard code to make it work or use wandb).   
2. Create a python script for evaluating models on images. 
3. Make Jupyter notebooks to show how to use different parts of pipelines with descriptions. 
4. Create documentation for the repository. 
5. Make description for most used functions 
6. Make a docker file or docker image for the repository. 
7. Make available models(some of them should be trained again). 
8. Make setup.py. 
9. Make requirements document.  
10. Correct train.py to make it work and where possible refactor code. 


<b> 5.Link to the GitHub repository</b>  
1) Github repository for the course - https://github.com/adasegroup/Deep-vectorization-P.R.
2) Official  Github repository for article Deep Vectorization of Technical Drawings  - https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings

<b>References:</b>  
1) V. Egiazarian, O. Voynov, A. Artemov, D. Volkhonskiy, A. Safin, M. Taktasheva, D. Zorin, and E. Burnaev.
Deep vectorization of technical drawings. arXiv preprint arXiv:2003.05471, 2020
2)  Wandb site - https://wandb.ai/site