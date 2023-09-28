# LF-EPICANet

In this work, I mainly explored the Epipolar Constraint Attention for light field image processing. 
And the key conclusion is that, when processing LF images, if we want to make better use of transformer with a limited network parameter, 
we can constrain the search range of "query" feature within the maxdisp range of the image, and the range appears like a sandglass.

Besides the maxdisp range, I tried many other constraint shapes by modifying the slope of sandglass or change the shape into a typical rectangle. Details can be seen in:
(not available now, once available, I will show the paper here.)
