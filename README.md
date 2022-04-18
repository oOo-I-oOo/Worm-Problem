# Worm-Problem

This program is a demonstration of some different techniques for trying to solve Moser's Worm problem, of which more information can be found at https://en.wikipedia.org/wiki/Moser%27s_worm_problem

Mosers Worm problem is a notoriously difficult problem in geometry, which asks what is the shape of smallest area that contains every unit length curve after translation and rotation. The difficulty of the problem cannot be overstated, and in general even the question of asking if a set contains every unit length curve is intractible. As such we should not expect any of our methods to converge to or yield the optimal answer.

The first hurdle to solving the question is that the number of unit length curves is infinite so we need to discretize the problem to be able to try to optimize. We do this by discretizing space into squares, and curves into polygonal segments, where angles are a multiple of some base angle. Now in this discretization, everything is finite and so we can describe curves and our containing set, the number of possibilities for both grow exponentionally making brute force search intractable. Thus we developed several methods to attempting to try to optimize, but before we optimize, we need a function.

We can frame the Worm Problem as an optimization problem as follows, minimize over all sets the area of that set plus the maximum over all curves of the minimum over all linear transformations of the lenth of the curve not lying in our set. Indeed, the optimal set will be a minima with value equal to that of the area of the set.

To find this minima, we attempt several methods.
