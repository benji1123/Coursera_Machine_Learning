**Overall Process **

This NN determines the **digit **pictured in a series of 20x20 px images **integers **in range(1,10).



1.  Forward Propagation
1.  Cost Function
1.  Gradient

**Forward Propagation**



---


<span style="text-decoration:underline;">What We Have</span>



*   **X **=** **design matrix
    *   **5,000** TE's (training examples)
    *   TE's = **400px **images; 400 inputs (one per pixel)

<table>
  <tr>
   <td rowspan="2" >
<strong>Training Example</strong>
<p>
<strong>Index</strong>
   </td>
   <td colspan="3" ><strong>Variables </strong>(value of input-features)
   </td>
   <td rowspan="2" ><strong>Y</strong>
<p>
(ground truths)
   </td>
  </tr>
  <tr>
   <td><strong>x<sub>1</sub></strong>
   </td>
   <td><strong>...</strong>
   </td>
   <td><strong>x<sub>400</sub></strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>Rand(1, 10)
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>Rand(1, 10)
   </td>
  </tr>
  <tr>
   <td>...
   </td>
   <td>...
   </td>
   <td>...
   </td>
   <td>...
   </td>
   <td>...
   </td>
  </tr>
  <tr>
   <td>500
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>stuff
   </td>
   <td>Rand(1, 10)
   </td>
  </tr>
</table>




*   **Initial Theta 1 **=  (25 x 401) matrix of weights. When applied to X, the first **hidden-layer **(a<sub>2</sub>) is computed. Note: the **required bias-layer** is not initially in X.
    *   sigmoid( X  * Thetat1<sup>T</sup> ) = a<sub>2</sub>
    *   Note: (X * Theta1<sup>T</sup>)<sup>  </sup>= 5000 x 25 matrix ⇒ 5000 x **26 **(bias units)
*   **Initial Theta 2 **= (10 x 26) matrix of weights. When applied to a<sub>2 </sub>, the **output layer **(y) is computed. Note: there is no bias layer in **Y; **hence, the "[...]" below:
    *   [ones(size(a2,1), 1)  a<sub>3</sub>]  * Theta2<sup>T </sup>= Y
    *   **Y **will be  (5000 x 10) matrix; for each of the 5000 rows, the **10 **columns represent one of the 10 potential classes (outcomes). Given a combination of (400) features, the NN gives a decimal-probability for the output (Y') being each of the 10 classes.

_Through this, we have the forward propagation-algorithm. _

**_Y can be computed for any given Theta. _**


<table>
  <tr>
   <td><strong>Formatting Y matrix</strong>
<p>
Converting Y from a probability matrix, to a <strong>Logical Matrix</strong>
<p>
if guess = 3 ⇒   0 0 1 0 0 0 0 0 0 0
<p>
yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
   </td>
   <td><strong>Cost Function</strong>
<p>
Computes <strong>overall error </strong>of Neural Network's prediction. 
<p>
<em>This differs from the individual <strong>deltas </strong>in backpropagation.</em>
<p>
J = (sum(sum((-yVec).*log(a3) - (1-yVec).*log(1-a3) )))/m;
   </td>
  </tr>
</table>


**Backpropagation **

Gradient Computation

-Computes a **(25 x 401)** matrix of Theta1 partial-derivatives (**gradient**) | same size as Theta1

-Computes **(10 x 26) **matrix of Theta2 partial-derivatives (**gradient**) | same size as Theta2



---



    **1 **- Compute "**ACCUMULATORS**" (coeff for Gradients)


    <span style="text-decoration:underline;">There is an _accumulator _for every **Theta-gradient**. Both associated with the same **layer**</span>



*   **Theta1** is the weight-matrix for computing **a<sub>2 </sub>**(hidden); **D1 **is the <span style="text-decoration:underline;">error of a<sub>2</sub></span> * a1
    *   A<sub>2 </sub>error = (d3*Theta2) .* [ones(size(z2,1),1) sigmoidGradient(z2)]; 
*   **Theta2** is the weight-matrix for computing **Y'**; **D2 **is the <span style="text-decoration:underline;">error of Y' </span>* a2   |   y' error = y' - y 

        D1 = d2(:,2:end)' * a1;    % coeff for Theta1_grad (bias term has no error)


        D2 = d3' * a2;             	 % coeff for Theta2_grad (y' has no bias term)


    **2 **- Compute Gradient (**not** **regularized**)


    **<span style="text-decoration:underline;">Gradients contribute to optimization of the next generation of Thetas.</span>**


    A simple equation if we do not want to **regularize **(yet)...


        Theta1_grad = (1/m)*D1;


        Theta2_grad = (1/m)*D2;


    **3 **- Regularize Gradient (**except bias-units**)


    <span style="text-decoration:underline;">Reg-terms are added to (**non-bias**) nodes. This is to facilitate more **generalization** and avoid **overfitting. **</span>


        Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*(Theta1(:,2:end));


        Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*(Theta2(:,2:end));


    **4 **- Unroll Gradient


    <span style="text-decoration:underline;">Combining **Thetas **into a **vector** (10,285 x 1). </span>


    This is input-form required by **optimization-algorithms**


    grad = [Theta1_grad(:) ; Theta2_grad(:)];
