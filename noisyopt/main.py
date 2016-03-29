import numpy as np
import scipy.stats

# include OptimizeResult class for machines on which scipy version is too old
try:
    from scipy.optimize import OptimizeResult
except:
    class OptimizeResult(dict):
        """ Represents the optimization result.
        Attributes
        ----------
        x : ndarray
            The solution of the optimization.
        success : bool
            Whether or not the optimizer exited successfully.
        status : int
            Termination status of the optimizer. Its value depends on the
            underlying solver. Refer to `message` for details.
        message : str
            Description of the cause of the termination.
        fun, jac, hess, hess_inv : ndarray
            Values of objective function, Jacobian, Hessian or its inverse (if
            available). The Hessians may be approximations, see the documentation
            of the function in question.
        nfev, njev, nhev : int
            Number of evaluations of the objective functions and of its
            Jacobian and Hessian.
        nit : int
            Number of iterations performed by the optimizer.
        maxcv : float
            The maximum constraint violation.
        Notes
        -----
        There may be additional attributes not listed above depending of the
        specific solver. Since this class is essentially a subclass of dict
        with attribute accessors, one can see which attributes are available
        using the `keys()` method.
        """
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __repr__(self):
            if self.keys():
                m = max(map(len, list(self.keys()))) + 1
                return '\n'.join([k.rjust(m) + ': ' + repr(v)
                                  for k, v in self.items()])
            else:
                return self.__class__.__name__ + "()"

#TODO: implement variable deltas for different directions (might speed up things, see review)

def minimize(func, x0, args=(), scaling=None, deltainit=1.0, deltatol=0.1, feps=1e-15,
            bounds=None, redfactor=2.0,
            errorcontrol=False, funcmultfactor=2.0, paired=False, alpha=0.05,
            disp=False, **kwargs):
    """
    Minimization of a function using a compass rule.

    Parameters
    ----------
    scaling:
        scaling by which to multiply step size and tolerances along different dimensions
    deltainit:
        inital pattern size
    deltatol:
        smallest pattern size
    redfactor:
        reduction factor by which to reduce delta if no reduction direction found 
    bounds:
        bounds on the variables
    errorcontrol:
        whether to control error of simulation 
    funcmultfactor: only for errorcontol=True
        multiplication factor by which to increase number of iterations of function
    paired: only for errorcontol=True
        compare for same random seeds
    alpha: only for errorcontol=True
        signficance level of tests
        (note: no correction for multiple testing thus interpret with care)

    Returns
    -------
    scipy.optimize.OptimizeResult object
    special entry: free
                   Boolean array indicating whether the variable is free (within feps) at the optimum
    """
    # absolute tolerance for float comparisons
    floatcompatol = 1e-14
    x0 = np.asarray(x0)
    if errorcontrol:
        func = AveragedFunction(func, paired=paired)
    else:
        func = memoized(func)
    if scaling is None:
        scaling = np.ones(x0.shape)
    else:
        scaling = np.asarray(scaling)
    if disp:
        print 'compass optimization starting'
        print 'args', args
        print 'errorcontrol', errorcontrol
        print 'paired', paired
    # ensure initial point lies within bounds
    if bounds is not None:
        np.clip(x0, bounds[:, 0], bounds[:, 1], out=x0)

    def clip(x, d):
        """clip x+d to respect bounds
        
        returns clipped result and effective distance"""
        xnew = x + d
        if bounds is not None:
            # if test point depasses set to boundary instead
            xclipped = np.clip(xnew, bounds[:, 0], bounds[:, 1])
            deltaeff = np.abs(x - xclipped).sum()
            return xclipped, deltaeff
        else:
            return xnew, delta
        
    # generate set of search directions (+- s_i e_i | i = 1, ...,  N)
    def unit(i, N):
        "return ith unit vector in R^N"
        arr = np.zeros(N)
        arr[i] = 1.0
        return arr
    N = len(x0)
    generatingset = [unit(i, N)*direction*scaling[i] for i in np.arange(N) for direction in [+1, -1]]

    x = x0 
    f = func(x0, *args)
    delta = deltainit
    # number of iterations:
    nit = 0
    finished = False
    while not finished:
        nit += 1
        # if delta gets close to deltatol, do iteration with step size deltatol instead
        # if no improvement possible then finished stays zero and algorithm is terminated
        # this ensures local optimality within deltatol
        if delta/redfactor < deltatol:
            delta = deltatol
            finished = True
        if disp:
            print 'nit %i, Delta %g' % (nit, delta)
        found = False
        np.random.shuffle(generatingset)
        for d in generatingset:
            xtest, deltaeff = clip(x, delta*d)
            if deltaeff == 0.0:
                continue
            ftest = func(xtest, *args)
            if (not errorcontrol and (ftest < f-feps)) or (errorcontrol and func.test(xtest, x, args, type_='smaller', alpha=alpha)):
                x = xtest
                f = ftest
                found = True
                if disp:
                    print x
                break
            elif ((deltaeff >= deltatol*np.sum(np.abs(d))) # do not try refinement for boundary steps smaller than tolerance
                    and  ((not errorcontrol and (ftest < f+feps))
                        or (errorcontrol
                            and func.test(xtest, x, args, type_='equality', alpha=alpha)
                            and (func.diffse(xtest, x, args) > feps)))):
                # If there is no significant difference the step size might
                # correspond to taking a step to the other side of the minimum.
                # Therefore test if middle point significantly better
                xmid = 0.5*(x+xtest)
                fmid = func(xmid, *args)
                if (not errorcontrol and fmid < f-feps) or (errorcontrol and func.test(xmid, x, args, type_='smaller')):
                    x = xmid
                    f = func(xmid, *args)
                    found = True
                    delta /= redfactor
                    if disp:
                        print 'mid', x
                    break
                # otherwise increase accuracy of simulation to try to get to significance
                elif errorcontrol:
                    func.setN(func.N * funcmultfactor)
                    if disp:
                        print 'new N %i' % func.N
                    if func.test(xtest, x, args, type_='smaller'):
                        x = xtest
                        f = ftest
                        found = True
                        if disp:
                            print x
                        break
                    # if we still have not resolved the differnce
                    finished = False
                    if disp:
                        print 'no significant difference yet', x, xtest, func.diffse(xtest, x, args)
        if not found:
            delta /= redfactor
        else:
            # optimization not finished if x updated during last iteration
            finished = False

    message = 'Succesful termination'
    # check if any of the directions are free at the optimum
    delta = deltatol
    free = np.zeros(x.shape, dtype=bool)
    for d in generatingset:
        dim = np.argmax(np.abs(d))
        xtest, deltaeff = clip(x, delta*d)
        if deltaeff < deltatol*np.sum(np.abs(d))-floatcompatol: # do not consider as free for boundary steps
            continue
        if not free[dim] and (((not errorcontrol and func(xtest) - feps < func(x)) or
            (errorcontrol and func.test(xtest, x, args, type_='equality', alpha=alpha)
                and (func.diffse(xtest, x, args) < feps)))):
            free[dim] = True
            message += '. dim %i is free at optimum' % dim
                
    reskwargs = dict(x=x, nit=nit, nfev=func.nev, message=message, free=free)
    if errorcontrol:
        res = OptimizeResult(fun=f[0], funse=f[1], **reskwargs)
    else:
        res = OptimizeResult(fun=f, **reskwargs)
    if disp:
        print res
    return res

class AverageBase(object):
    """
    Base class for averaged evaluation of noisy functions.
    """
    def __init__(self, N=30, paired=False):
        """
        N: number of calls to average over.
        paired: if paired is chosen the same series of random seeds is used for different x
        """
        self.N = N
        self.paired = paired
        if self.paired:
            self.uint32max = np.iinfo(np.uint32).max 
            self.seeds = list(np.random.randint(0, self.uint32max, size=N))
        # cache previous iterations
        self.cache = {}
        # number of evaluations
        self.nev = 0

    def setN(self, N):
        N = int(N)
        if self.paired and (N > self.N):
            Nadd = N - self.N
            self.seeds.extend(list(np.random.randint(0, self.uint32max, size=Nadd)))
        self.N = N

class AveragedFunction(AverageBase):
    """Averages a function's return value over a number of runs

        func(x, *args)

        Caches previous results.
    """
    def __init__(self, func, **kwargs):
        super(AveragedFunction, self).__init__(**kwargs)
        self.func = func

    def __call__(self, x, *args):
        #convert to tuple (hashable!)
        xt = tuple(x)
        if xt in self.cache:
            Nold = len(self.cache[xt])
            if Nold < self.N:
                Nadd = self.N - Nold 
                if self.paired:
                    values = [self.func(x, *args, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                else:
                    values = [self.func(x, *args) for i in range(Nadd)]
                self.cache[xt].extend(values)
                self.nev += Nadd
        else:
            if self.paired:
                values = [self.func(x, *args, seed=self.seeds[i]) for i in range(self.N)]
            else:
                values = [self.func(x, *args) for i in range(self.N)]
            self.cache[xt] = values 
            self.nev += self.N
        return np.mean(self.cache[xt]), np.std(self.cache[xt], ddof=1)/self.N**.5

    def diffse(self, xtest, x, args=()):
        """Standard error of the difference between the function values at x and xtest""" 
        ftest, ftestse = self(xtest, *args)
        f, fse = self(x, *args)
        if self.paired:
            fxtest = np.array(self.cache[tuple(xtest)])
            fx = np.array(self.cache[tuple(x)])
            diffse = np.std(fxtest-fx, ddof=1)/self.N**.5 
            return diffse
        else:
            return (ftestse**2 + fse**2)**.5

    def test(self, xtest, x, args=(), alpha=0.05, type_='smaller'):
        """ type in ['smaller', 'equality']."""
        # call function to make sure it has been evaluated a sufficient number of times
        if type_ not in ['smaller', 'equality']:
            raise NotImplementedError(type_)
        ftest, ftestse = self(xtest, *args)
        f, fse = self(x, *args)
        # get function values
        fxtest = np.array(self.cache[tuple(xtest)])
        fx = np.array(self.cache[tuple(x)])
        if np.mean(fxtest-fx) == 0.0:
            if type_ == 'equality':
                return True
            if type_ == 'smaller':
                return False
        if self.paired:
            # if values are paired then test on distribution of differences
            statistic, pvalue = scipy.stats.ttest_rel(fxtest, fx, axis=None)
        else:
            statistic, pvalue = scipy.stats.ttest_ind(fxtest, fx, equal_var=False, axis=None)
        if type_ == 'smaller':
            # if paired then df=N-1, else df=N1+N2-2=2*N-2 
            df = self.N-1 if self.paired else 2*self.N-2
            pvalue = scipy.stats.t.cdf(statistic, df) 
            # return true if null hypothesis rejected
            return pvalue < alpha
        if type_ == 'equality':
            # return true if null hypothesis not rejected
            return pvalue > alpha

class DifferenceFunction(AverageBase):
    """Averages the difference of two function's return values over a number of runs
    """
    def __init__(self, func1, func2, **kwargs):
        super(DifferenceFunction, self).__init__(**kwargs)
        self.funcs = [func1, func2]

    def __call__(self, x, *args):
        try:
            # convert to tuple (hashable!)
            xt = tuple(x)
        except TypeError:
            # if TypeError then likely floating point value
            xt = (x, )
        for i, func in enumerate(self.funcs):
            ixt = i, xt
            if ixt in self.cache:
                Nold = len(self.cache[ixt])
                if Nold < self.N:
                    Nadd = self.N - Nold 
                    if self.paired:
                        values = [func(x, *args, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                    else:
                        values = [func(x, *args) for i in range(Nadd)]
                    self.cache[ixt].extend(values)
                    self.nev += Nadd
            else:
                if self.paired:
                    values = [func(x, *args, seed=self.seeds[i]) for i in range(self.N)]
                else:
                    values = [func(x, *args) for i in range(self.N)]
                self.cache[ixt] = values 
                self.nev += self.N
        diff = np.asarray(self.cache[(0, xt)]) - np.asarray(self.cache[(1, xt)])
        return np.mean(diff), np.std(diff, ddof=1)/self.N**.5

    def test(self, x, args=(), alpha=0.05, type_='smaller'):
        """ type_ in ['smaller', 'equality']."""
        diff, diffse = self(x, *args)
        epscal = diff / diffse
        if type_ == 'smaller':
            return epscal < scipy.stats.norm.ppf(alpha)
        if type_ == 'equality':
            return np.abs(epscal) < scipy.stats.norm.ppf(1-alpha/2.0)
        raise NotImplementedError(type_)

    def testtruesmaller(self, x, *args, **kwargs):
        kwargs['args'] = args
        kwargs['type_'] = 'equality'
        disp = kwargs.pop('disp', False)
        feps = kwargs.pop('feps', 1e-15)
        while self.test(x, **kwargs) and self(x, *args)[1] > feps:
            self.setN(self.N*2.0)
            if disp:
                print 'testtruesmaller', self.N, self(x, *args)[1]
        kwargs['type_'] = 'smaller'
        return self.test(x, **kwargs)

def bisect(func, a, b, args=(), xtol=1e-6, errorcontrol=False, alpha=0.05, disp=False):
    """Find root by bysection search.

    Parameters
    ----------
    a, b:
        initial interval
    args:
        extra args to be supplied to function
    xtol:
        target tolerance for intervall size
    errorcontrol:
        if true, assume that function is instance of DifferenceFunction  
    alpha: (only for errorcontrol=True)
        significance level to be used for testing 

    Returns
    -------
    root of function
    """
    width = b-a 
    # check whether function is ascending or not
    if errorcontrol:
        testkwargs = dict(alpha=alpha, disp=disp)
        fa = func.testtruesmaller(a, *args, **testkwargs)
        fb = func.testtruesmaller(b, *args, **testkwargs)
    else:
        fa = func(a, *args) < 0
        fb = func(b, *args) < 0
    if fa and not fb:
        ascending = True
    elif fb and not fa:
        ascending =  False
    else:
        print 'Warning: func(a) and func(b) do not have opposing signs -> no search done'
        width = 0.0

    while width > xtol:
        mid = (a+b)/2.0
        if ascending:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.testtruesmaller(mid, *args, **testkwargs)):
                a = mid 
            else:
                b = mid
        else:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.testtruesmaller(mid, *args, **testkwargs)):
                b = mid 
            else:
                a = mid
        if disp:
            print 'bisect bounds', a, b
        width /= 2.0
    # interpolate linearly to get zero
    if errorcontrol:
        ya, yb = func(a, *args)[0], func(b, *args)[0]
    else:
        ya, yb = func(a, *args), func(b, *args)
    m = (yb-ya) / (b-a)
    res = a-ya/m
    if disp:
        print 'bisect final value', res
    return res

class memoized(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    
    can be turned of by passing memoize=False when calling the function
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.nev = 0

    def __call__(self, *args, **kwargs):
        # if args is not Hashable we can't cache
        # easier to ask for forgiveness than permission
        memoize = kwargs.pop('memoize', True)
        if memoize:
            try:
                index = ()
                for arg in args:
                    index += tuple(arg)
                # try to also recompute if kwargs changed
                for item in kwargs.itervalues():
                    try:
                        index += (float(item), )
                    except:
                        pass
                if index in self.cache:
                    return self.cache[index]
                else:
                    value = self.func(*args, **kwargs)
                    self.nev += 1
                    self.cache[index] = value
                    return value
            except TypeError:
                print 'not hashable', args
                self.nev += 1
                return self.func(*args, **kwargs)
        else:
            self.nev += 1
            return self.func(*args, **kwargs)

if __name__ == "__main__":
    def quadratic(x):
        return (x**2).sum()
    print compass(quadratic, np.asarray([0.5, 1.0]))
    print compass(quadratic, np.asarray([2.5, -3.2]))
    print compass(quadratic, np.asarray([2.5, -3.2, 0.9, 10.0, -0.3]))
    print compass(quadratic, np.asarray([0.5, 0.5]), bounds=np.asarray([[0, 1], [0, 1]]))
    print compass(quadratic, np.asarray([0.8, 0.8]), bounds=np.asarray([[0.5, 1], [0.5, 1]]), deltatol=0.01)

    import scipy.optimize
    print compass(scipy.optimize.rosen, np.asarray([-3.0, -4.0]), deltatol=0.00001)

#    import evolimmun
#    lambda_ = 3
#    mu = 1
#    aenv = 0.1
#    pienv = 0.1
#    Delta = 0.8
#    niter = 1e5
#    nburnin = 1e3
#    g = lambda x: 2*x/(1+x)
#    k = lambda x: 0.1*x+x**2
#    bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
#    args = (lambda_, mu, aenv, pienv, Delta, niter, nburnin, g, k)
#    print compass(evolimmun.minus(evolimmun.Lambda_pq), [0.1, 0.0, 1.0, 0.0],
#                   scaling=(1.0, 1.0, 5.0, 1.0),
#                   args=args, bounds=bounds,
#                   deltainit=0.1,
#                   deltatol=0.01,
#                   errorcontrol=True,
#                   paired=True,
#                   disp=False)

    def matya(x):
        return 0.26*(x[0]**2 + x[1]**2)-0.48*x[0]*x[1]
    print compass(matya, np.asarray([2.0, 3.5]), deltatol=0.01)

    def stochastic_quadratic(x, seed=None):
        prng = np.random if seed is None else np.random.RandomState(seed)
        return (x**2).sum() + prng.randn(1) + 0.5*np.random.randn(1)
   
    diff = DifferenceFunction(stochastic_quadratic, stochastic_quadratic)
    print diff(np.array([1.0, 2.0]))
    print diff.test(np.array([1.0, 2.0]), (), type_ = 'equality')

    print compass(stochastic_quadratic, np.array([4.55, 3.0]), deltainit=2.5, deltatol=0.4, errorcontrol=True)
    print 'paired', compass(stochastic_quadratic, np.array([4.55, 3.0]), deltainit=2.5, deltatol=0.4, errorcontrol=True, paired=True)
