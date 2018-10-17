import re
import calendar
from string import ascii_lowercase as alphabet
import os
import glob
import numpy as np
import json
import datetime
import warnings
from scicatalog.export import _tex_fmt as tex_fmt

def _strip_braces(line):
    while line.endswith('}') and line.startswith('{'):
        line = line[1:-1]
    return line


class BibEntry(object):
    # technically, I should make this object immutable since I want it to be hashable, but I'm going to be lazy and
    # reckless and not do it
    def __init__(self, text):
        # get rid of anything after the last } and before the @
        reverse = text[::-1]
        i = reverse.find('}')
        text = reverse[i:][::-1]
        i = text.find('@')
        text = text[i:]

        self.text = text
        if self.doi is not None:
            self.id = self.doi
        elif self.publoc is not None:
            self.id = self.publoc
        else:
            self.id = self.text

    def get_line(self, key):
        match_str = r'{} *= *([^=]*?)(,\s*?\n\s*\w*\s*=|,?\r?\n?}}$)'.format(key)
        match = re.search(match_str, self.text)
        if match is None:
            return None
        line = match.group(1)
        line = _strip_braces(line)
        line = re.sub('[ \n]+', ' ', line)
        return line

    authors = property(lambda self: self.get_line('author'))
    journal = property(lambda self: self.get_line('journal'))
    volume = property(lambda self: self.get_line('volume'))
    pages = property(lambda self: self.get_line('pages'))
    doi = property(lambda self: self.get_line('doi'))
    month = property(lambda self: self.get_line('month'))
    year = property(lambda self: self.get_line('year'))

    def get_publoc(self):
        pieces = [self.journal, self.volume, self.pages]
        if any([piece is None for piece in pieces]):
            return None
        else:
            return ' '.join(pieces)
    publoc = property(get_publoc)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def std_key(self):
        """Automatically generates a lastnameYY key for the bibtex entry, where YY is the last two digits of the
        publication year.

        In generating the lastname, the following rules are obeyed:
        - The lastname of the first author is used.
        - The name is entirely lowercase.
        - In hyphenated names or names with spaces, only the portion before the hyphen is used.
        - For names with special characters (e.g. accented), the character is converted to the analagous "normal" character.
        - In names beginning with ['Von', 'De'] or any word ending with a period, that first word is dropped.
        """

        # parse last name
        authorline = self.authors
        if authorline is None:
            authorline = self.get_line('editor')
        lastname = authorline.split(',')[0]
        lastname = _strip_braces(lastname)

        # make lowercase
        lastname = lastname.lower()

        # remove throwaway articles
        toss = ['von', 'de', 'van']
        for article in toss:
            lastname = re.sub(r'(?<!\w)' + article + ' ', '', lastname)

        # convert special characters
        lastname = re.sub(r'{\\.*?([a-zA-Z])}', lambda match: match.group(1), lastname)

        # keep only start if compound
        for s in ('-', ' '):
            lastname = lastname.split(s)[0]

        # remove any remaining weird characters
        lastname = re.sub(r'[^a-zA-Z]', '', lastname)

        # parse year
        yy = self.year[-2:]

        return lastname + yy

    def __str__(self):
        return self.text


def mergeBibs(input_dir, savepath):
    """Load all the .bbl files in dir and combine them all into a single .bbl with lastnameYY keys."""
    paths = glob.glob(os.path.join(input_dir, '*.bbl'))
    paths += glob.glob(os.path.join(input_dir, '*.bib'))
    bibstr = ''
    for path in paths:
        with open(path) as stream:
            bibstr += stream.read()

    entries = parseBibEntries(bibstr)

    # remove duplicates
    entries = list(set(entries))

    entries = rekeyBib(entries)

    bibstr = '\n\n'.join(map(str, entries)) + '\n'

    # apparently ADS sometimes likes to put \# in the entries. clean those out
    bibstr = bibstr.replace('\\#', '')
    bibstr = bibstr.replace('#', '')

    with open(savepath, 'w') as f:
        f.write(bibstr)


def parseBibEntries(bibstr):
    """Returns a list of individual bibtex entries when given a single string containing them."""

    entries = bibstr.split('@')[1:]
    entries = [BibEntry('@' + s.strip()) for s in entries]

    return entries


def rekeyBib(entryList):
    """
    Generates keys for an entire list of bibtex entries and inserts them into the entries themsleves.

    If two keys would otherwise be identical, 'a', 'b', 'c', etc. are added to the end, where the letters sort the
    entries according to the month of publication.
    """

    # first pass, just generate and store all the keys
    keys = [entry.std_key() for entry in entryList]

    # zip and sort entries by the key
    pairs = zip(entryList, keys)
    sortkey = lambda pair: pair[1]
    pairs = sorted(pairs, key=sortkey)

    # loop through and add 'a', 'b', ... to duplicates, sorted by month of publication
    i = 0
    while i < (len(pairs) - 1):
        j = i + 1
        while j < len(pairs) and pairs[i][1] == pairs[j][1]:
            j += 1
        if (j-i) == 1:
            i += 1
            continue
        else:
            group = pairs[i:j]
            group = sorted(group, key=_get_month)
            for k, (entry, key) in enumerate(group):
                key += alphabet[k]
                group[k] = (entry, key)
            pairs[i:j] = group

    # insert the keys into the appropriate spot
    for entry, key in pairs:
        replace = lambda match: '@{}{{{},'.format(match.group(1), key)
        entry.text = re.sub(r'@([A-Z]*)\{(.*),', replace, entry.text)

    return zip(*pairs)[0]


_month_abbrs = [month.lower() for month in calendar.month_abbr]
def _get_month(pair):
        entry = pair[0]
        month_abbr = entry.month
        if month_abbr is None:
            return -1
        else:
            return _month_abbrs.index(month_abbr.lower())


class Results(dict):
    """Basic class to keep numerical results for a paper in one dictionary-like object that is automatically saved to
     the disk when values are updated."""
    dict_name = 'results_dictionary.json'

    def __init__(self, path):
        """Create an empty results object at the path indicated or load if the path already exists."""
        self.dir = path
        if os.path.exists(path):
            with open(os.path.join(path, self.dict_name)) as f:
                dic = json.load(f)
            for key, value in dic.items():
                super(Results, self).__setitem__(key, value)
        else:
            os.mkdir(path)
            self.save_dict()

    def save_dict(self):
        with open(os.path.join(self.dir, self.dict_name), 'w') as f:
            json.dump(self, f)

    def __setitem__(self, key, value):
        warnings.warn('Dictionary-style setting uses default values for string export settings. '
                      'Use set method if you want to customize the string representation of the value.')
        self.set(key, value)

    def upgrade_old_results(self):
        for key in self.keys():
            value = super(Results, self).__getitem__(key)
            self.set(key, value, save=False)
            self.save_dict()

    def _make_entry(self, value, sigfigs=2, forceSN=False):
        time_string = datetime.datetime.now().isoformat()
        time_string = time_string.split('.')[0]
        return dict(value=value, updated=time_string, sigfigs=sigfigs, forceSN=forceSN)

    def set(self, key, value=None, sigfigs=2, force_sci_notation=False, save=True):
        if value is None:
            if key not in self:
                raise ValueError('Must specify a value unless key is already in Results.')
            else:
                value = self[key]
        if np.array(value).size > 3:
            pathname = os.path.join(self.dir, key + '.npy')
            np.save(pathname, np.array(value))
            value = pathname
        entry = self._make_entry(value, sigfigs=sigfigs, forceSN=force_sci_notation)
        super(Results, self).__setitem__(key, entry)
        if save:
            self.save_dict()

    def export_sty(self, keys=None, path='numbers.sty'):
        """
        Export a latex style file with the numbers as entries.

        Each number is given a command based on the key used for it in the results dictionary. Non latex-allowed
        characters are mapped as follows:
            _ -> removed and next letter capitalized
            number -> changed to roman numeral if < 100, else removed

        Values that are length three lists are assumed to be values with
        error bars and five separate latex macros are made accordingly
            key = just the value
            keyPosErr = positive error
            keyNegErr = negative error
            keyErr = average of positive and negative errors
            keyTrip = value with error bars

        Values are not printed in math mode, i.e. bracketing $ characters
        are omitted and must be manually added if the value is printed in
        scientific notation or with error bars.

        Parameters
        ----------
        keys : keys for values to export, None exports all
        path : path for the exported style file

        Returns
        -------
        None
        """
        lines = []

        def add_line(key, value, date=None):
            line = '\\newcommand{{{}}}{{{}}}'.format(key, value)
            if date:
                line += ' % {}'.format(date)
            lines.append(line)

        for key in keys:
            entry = self[key]
            value = entry['value']
            forceSN = entry['forceSN']
            sigfigs = entry['sigfigs']
            if hasattr(value, 'size') and value.size > 3:
                print '{} skipped because value is an array.'.format(key)

            # reformat key to be tex allowable
            # get rid of underscores
            tex_key = re.sub('_+', '_', key)
            while True:
                i = tex_key.find('_')
                if i == -1:
                    break
                else:
                    if i < len(tex_key) - 1:
                        next_char = tex_key[i+1]
                        tex_key = tex_key[:i] + next_char.upper() + tex_key[i+2:]

            # replace numbers
            match = re.findall('\d+', tex_key)
            if match is not None:
                for num in match:
                    i = tex_key.find(num, 1)
                    tex_key = tex_key[:i] + num2roman(int(num)) + tex_key[i+len(num):]

            # get tex-formatted value
            if forceSN:
                fmt = '{{:.{}e}}'.format(sigfigs-1)
            else:
                fmt = '{{:.{}g}}'.format(sigfigs)
            val_str = tex_fmt(value, None, None, sigfigs, fmt, forceSN)
            add_line(tex_key, val_str, entry['updated'])
            if len(value) == 3:
                errneg = abs(value[1])
                errpos = abs(value[2])
                full_str = tex_fmt(value[0], errneg, errpos, sigfigs, fmt, forceSN)
                add_line(tex_key + 'Trip', full_str)
                efmt = '{{:.{}e}}'.format(sigfigs-1)
                NegErr = tex_fmt(errneg, None, None, sigfigs, efmt)
                PosErr = tex_fmt(errpos, None, None, sigfigs, efmt)
                Err = tex_fmt((errneg + errpos) / 2, None, None, sigfigs, efmt)
                add_line(tex_key + 'NegErr', NegErr)
                add_line(tex_key + 'PosErr', PosErr)
                add_line(tex_key + 'Err', Err)

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


    def __getitem__(self, item):
        entry = super(Results, self).__getitem__(item)
        entry = entry.copy()
        if str(entry['value']) == entry['value'] and os.path.exists(entry['value']):
            ary = np.load(entry['value'])
            entry['value'] = ary
        return entry

    def __delitem__(self, key):
        value = super(Results, self).__getitem__(key)
        if isinstance(value, basestring) and os.path.exists(value):
            os.remove(value)
        super(Results, self).__delitem__(key)
        self.save_dict()


def number_check(ms_path):
    with open(ms_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for m in re.finditer('([^a-zA-Z]*\\d+[^a-zA-Z]*)+', line):
            i0 = m.start() - 20
            if i0 < 0:
                i0 = 0
            i1 = m.end() + 20
            snippet = line[i0:i1]
            snippet = '{}: '.format(i) + snippet
            print snippet


_num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
            (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
def num2roman(num):
    roman = ''
    while num > 0:
        for i, r in _num_map:
            while num >= i:
                roman += r
                num -= i
    return roman