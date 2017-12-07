import re
import calendar
from string import ascii_lowercase as alphabet
import os
import glob

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
            return _month_abbrs.index(month_abbr)




