from astropy import units as u, constants as const
from warnings import warn


class Line(object):
    def __init__(self, name, w, A, f, m, label=None, AASTex_label=None):
        self.name = name
        self.w = w
        self.A = A
        self.f = f
        self.m = m
        self.label = label
        self.AASTex_label = AASTex_label

    def __repr__(self):
        return ('{} | w = {:.2f} | A = {:.1e} | f = {:.3f} | m = {:.1f}'
                ''.format(self.label, self.w, self.A, self.f, self.m))

    @property
    def nu(self):
        return (const.c/self.w).to('Hz')


class Multiplet(object):
    def __init__(self, name, lines, label=None, AASTex_label=None):
        # have to set params like this or setattr is infinite recursion loop
        self.__dict__['name'] = name
        self.__dict__['label'] = label
        self.__dict__['lines'] = lines
        self.__dict__['AASTex_label'] = AASTex_label

    def __getattr__(self, item):
        lst = [getattr(line, item) for line in self.lines]
        if hasattr(lst[0], 'unit'):
            unit = lst[0].unit
            lst = [item.value for item in lst] * unit
        return lst

    def __setattr__(self, key, value):
        if key == 'lines':
            raise ValueError('Can only set lines attribute on '
                             'initialization.')
        if hasattr(value, '__iter__') and len(value) == len(self.lines):
                for line, val in zip(self.lines, value):
                    setattr(line, key, val)
        else:
            raise ValueError('Input could not be shaped to match lines.')

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def __getitem__(self, item):
        if type(item) is int:
            return self.lines[item]
        elif type(item) is slice:
            return Multiplet(self.name, self.lines[item], self.label, self.AASTex_label)
        else:
            raise KeyError('Can only index lines with int or slice.')

    def __repr__(self):
        return '\n'.join([self.label] + ['-'*len(self.label)] +
                         [repr(line) for line in self.lines])

    def reduce(self):
        warn('I do not know if this is how you combine a multiplet into a singlet. Check!!')
        A = sum(self.A)
        f = sum(self.f)
        w = sum(self.f*self.w)/sum(self.f)
        return Line(self.name, w, A, f, self.m[0], self.label, self.AASTex_label)


c2 = Multiplet('c2', label='C II', AASTex_label='\ion{C}{2}',
               lines = (Line('c2a', 1334.532*u.AA, 2.41e+08*u.Hz, 0.129, 12.011*u.u),
                        Line('c2b', 1335.708*u.AA, 2.88e+08*u.Hz, 0.115, 12.011*u.u)))

c3 = Multiplet('c3', label='C III', AASTex_label='\ion{C}{3}',
               lines = (Line('c3a', 1174.933*u.AA, 3.293e+08*u.Hz, 1.136e-01, 12.011*u.u),
                        Line('c3b', 1175.263*u.AA, 4.385e+08*u.Hz, 2.724e-01, 12.011*u.u),
                        Line('c3c', 1175.59*u.AA, 3.287e+08*u.Hz, 6.810e-02, 12.011*u.u),
                        Line('c3d', 1175.711*u.AA, 9.856e+08*u.Hz, 2.042e-01, 12.011*u.u),
                        Line('c3e', 1175.987*u.AA, 1.313e+09*u.Hz, 9.074e-02, 12.011*u.u),
                        Line('c3f', 1176.370*u.AA, 5.468e+08*u.Hz, 6.807e-02, 12.011*u.u)))

c4 = Multiplet('c4', label='C IV', AASTex_label='\ion{C}{IV}',
               lines = (Line('c4a', 1548.202*u.AA, 2.65e+08*u.Hz, 1.90e-01, 1.90e-01*u.u),
                        Line('c4b', 1550.774*u.AA, 2.64e+08*u.Hz, 9.52e-02, 9.52e-02*u.u)))

he2 = Multiplet('he2', label='He II', AASTex_label='\ion{He}{2}',
                lines = (Line('he2a', 1640.33212826*u.AA, 8.6259e+08*u.Hz, 6.9591e-01, 4.00260*u.u),
                         Line('he2b', 1640.3446542 *u.AA, 3.5939e+08*u.Hz, 2.8995e-01, 4.00260*u.u),
                         Line('he2c', 1640.37499364*u.AA, 3.3693e+07*u.Hz, 1.3592e-02, 4.00260*u.u),
                         Line('he2d', 1640.3913521*u.AA, 3.5939e+08*u.Hz, 1.3592e-02, 4.00260*u.u),
                         Line('he2e', 1640.47417469*u.AA, 1.0349e+09*u.Hz, 6.2629e-01, 4.00260*u.u),
                         Line('he2f', 1640.48974134*u.AA, 1.7248e+08*u.Hz, 6.9588e-02, 4.00260*u.u),
                         Line('he2g', 1640.53261496*u.AA, 6.7379e+07*u.Hz, 1.3593e-02, 4.00260*u.u)))

mg2 = Multiplet('mg2', label='Mg II', AASTex_label='\ion{Mg}{2}',
                lines = (Line('mg2a', 2796.352*u.AA, 2.60e+08*u.Hz, 6.08e-01, 24.305*u.u),
                         Line('mg2b', 2803.531*u.AA, 2.57e+08*u.Hz, 3.03e-01, 24.305*u.u)))

n5 = Multiplet('n5', label='N V', AASTex_label='\ion{N}{5}',
               lines = (Line('n5a', 1238.821*u.AA, 3.40e+08*u.Hz, 1.56e-01, 14.0067*u.u),
                        Line('n5b', 1242.804*u.AA, 3.37e+08*u.Hz, 7.80e-02, 14.0067*u.u)))

si3 = Multiplet('si3', label='Si III', AASTex_label='\ion{Si}{3}',
                lines = (Line('si3', 1206.51*u.AA, 2.55e+09*u.Hz, 1.67e+00, 28.0855*u.u),))

si4 = Multiplet('si4', label='Si IV', AASTex_label='\ion{Si}{4}',
                lines = (Line('si4a', 1393.755*u.AA, 8.80e+08*u.Hz, 5.13e-01, 28.0855*u.u),
                         Line('si4b', 1402.770*u.AA, 8.63e+08*u.Hz, 2.55e-01, 28.0855*u.u)))

line_dict = {'c2': c2,
             'c3': c3,
             'c4': c4,
             'he2': he2,
             'mg2': mg2,
             'n5': n5,
             'si3': si3,
             'si4': si4}