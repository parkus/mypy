import re
import calendar
from string import ascii_lowercase as alphabet
import os

def mergeBibs(input_dir, savepath):
    """Load all the .bbl files in dir and combine them all into a single .bbl with lastnameYY keys."""
    names = filter(lambda s: s.endswith('.bbl'), os.listdir(input_dir))
    paths = [os.path.join(input_dir, name) for name in names]
    bibstr = ''
    for path in paths:
        with open(path) as stream:
            bibstr += stream.read()

    entries = parseBibEntries(bibstr)

    # remove duplicates
    entries = list(set(entries))

    entries = rekeyBib(entries)

    bibstr = '\n\n'.join(entries) + '\n'

    # apparently ADS sometimes likes to put \# in the entries. clean those out
    bibstr = bibstr.replace('\\#', '')
    bibstr = bibstr.replace('#', '')

    with open(savepath, 'w') as f:
        f.write(bibstr)


def parseBibEntries(bibstr):
    """Returns a list of individual bibtex entries when given a single string containing them."""

    entries = bibstr.split('@')[1:]
    entries = ['@' + s.strip() for s in entries]

    return entries


def rekeyBib(entryList):
    """
    Generates keys for an entire list of bibtex entries and inserts them into the entries themsleves.

    If two keys would otherwise be identical, 'a', 'b', 'c', etc. are added to the end, where the letters sort the
    entries according to the month of publication.
    """

    # first pass, just generate and store all the keys
    keys = map(bibEntryKey, entryList)

    # zip and sort entries by the key
    pairs = zip(entryList, keys)
    sortkey = lambda pair: pair[1]
    pairs = sorted(pairs, key=sortkey)

    # loop through and add 'a', 'b', ... to duplicates, sorted by month of publication
    i = 0
    while i < (len(pairs) - 1):
        j = i + 1
        while pairs[i][1] == pairs[j][1]:
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
    entries = []
    for entry, key in pairs:
        replace = lambda match: '@{}{{{},'.format(match.group(1), key)
        entry = re.sub(r'@([A-Z]*)\{(.*),', replace, entry)
        entries.append(entry)

    return entries


_month_abbrs = [month.lower() for month in calendar.month_abbr]
def _get_month(pair):
        entry = pair[0]
        match = re.search(r'month = (\w*),', entry)
        month_abbr = match.group(1)
        return _month_abbrs.index(month_abbr)


def bibEntryKey(entry):
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
    start = entry.index('{{') + 2
    stop = entry.index('},')
    lastname = entry[start:stop]

    # split at whitespace, keep first piece not matching a throwaway article
    toss = ['von', 'de', 'van']
    pieces = lastname.split()
    keep = 0
    while (pieces[keep][-1] == '.') or (pieces[keep].lower() in toss):
        keep += 1
    lastname = pieces[keep]

    # keep only start if hyphenated
    pieces = lastname.split('-')
    lastname = pieces[0]

    # convert special characters
    replace = lambda match: match.group(1)
    lastname = re.sub(r'\{(.)*\}', replace, lastname)

    # make lowercase
    lastname = lastname.lower()

    # parse year
    match = re.search(r'year = \d\d(\d\d),', entry)
    yy = match.group(1)

    return lastname + yy

