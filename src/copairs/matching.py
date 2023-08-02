'''
Sample pairs with given column restrictions
'''
import itertools
from collections import namedtuple
import logging
from math import comb
from typing import Sequence, Set, Union, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger('copairs')
ColumnList = Union[Sequence[str], pd.Index]
ColumnDict = Dict[str, ColumnList]


def reverse_index(col: pd.Series) -> pd.Series:
    '''Build a reverse_index for a given column in the DataFrame'''
    return pd.Series(col.groupby(col).indices, name=col.name)


def dict_to_dframe(dict_pairs, sameby: Union[str, list]):
    '''Convert the Matcher.get_all_pairs output to pd.DataFrame'''
    if not dict_pairs:
        raise ValueError('dict_pairs empty')
    keys = np.array(list(dict_pairs.keys()))
    counts = [len(pairs) for pairs in dict_pairs.values()]
    keys = np.repeat(keys, counts, axis=0)

    if keys.ndim > 1:
        # is a ComposedKey
        keys_df = pd.DataFrame(keys)  #, columns=sameby)
    else:
        if isinstance(sameby, list):
            sameby = sameby[0]
        keys_df = pd.DataFrame({sameby: keys})

    # Concat all pairs
    pairs_ix = itertools.chain.from_iterable(dict_pairs.values())
    pairs_df = pd.DataFrame(pairs_ix, columns=['ix1', 'ix2'])
    return pd.concat([keys_df, pairs_df], axis=1)


class UnpairedException(Exception):
    '''Exception raised when a row can not be paired with any other row in the
    data'''


class Matcher():
    '''Class to get pair of rows given contraints in the columns'''

    def __init__(self,
                 dframe: pd.DataFrame,
                 columns: ColumnList,
                 seed: int,
                 max_size: Optional[int] = None):
        '''
        max_size: max number of rows to consider from the same value.
        '''
        rng = np.random.default_rng(seed)
        dframe = dframe[columns].reset_index(drop=True).copy()
        dframe.index.name = '__copairs_ix'

        def clip_list(elems: np.ndarray) -> np.ndarray:
            if max_size and elems.size > max_size:
                logger.warning(f'Sampling {max_size} out of {elems.size}')
                elems = rng.choice(elems, max_size)
            return elems

        mappers = [
            reverse_index(dframe[col]).apply(clip_list) for col in dframe
        ]

        # Create a column order based on the number of potential row matches
        # Useful to solve queries with more than one sameby
        n_pairs = {}
        for mapper in mappers:
            n_combs = mapper.apply(lambda x: comb(len(x), 2)).sum()
            n_pairs[mapper.name] = n_combs
        col_order = sorted(n_pairs, key=n_pairs.get)
        self.col_order = {column: i for i, column in enumerate(col_order)}

        self.values = dframe[columns].values
        self.reverse = {
            mapper.name: mapper.apply(set).to_dict()
            for mapper in mappers
        }
        self.rng = rng
        self.frozen_valid = frozenset(range(len(self.values)))
        self.col_to_ix = {c: i for i, c in enumerate(columns)}
        self.columns = columns
        self.n_pairs = n_pairs
        self.rand_iter = iter([])

    def _null_sample(self, diffby_all: ColumnList, diffby_any: ColumnList):
        '''
        Sample a pair from the frame.
        '''
        valid = set(self.frozen_valid)
        id1 = self.integers(0, len(valid) - 1)
        valid.remove(id1)
        valid = self._filter_diffby(id1, diffby_all, diffby_any, valid)

        if len(valid) == 0:
            # row1 = self.values[id1]
            # assert np.any(row1 == self.values, axis=1).all()
            raise UnpairedException(f'{id1} has no pairs')
        id2 = self.choice(list(valid))
        return id1, id2

    def sample_null_pair(self, diffby: ColumnList, n_tries=5):
        '''Sample pairs from the data. It tries multiple times before raising an error'''
        if isinstance(diffby, dict):
            diffby_all, diffby_any = diffby.get("all", []), diffby.get("any", [])
            if len(diffby_any) == 1:
                raise ValueError('diffby: any should have more than one column')
        else:
            diffby_all = [diffby] if isinstance(diffby, str) else diffby
            diffby_any = []
        
        for _ in range(n_tries):
            try:
                return self._null_sample(diffby_all, diffby_any)
            except UnpairedException:
                pass
        raise ValueError(
            'Number of tries exhusted. Could not find a valid pair')

    def rand_next(self):
        try:
            value = next(self.rand_iter)
        except StopIteration:
            rands = self.rng.uniform(size=int(1e6))
            self.rand_iter = iter(rands)
            value = next(self.rand_iter)
        return value

    def integers(self, min_val, max_val):
        return int(self.rand_next() * (max_val - min_val + 1) + min_val)

    def choice(self, items):
        min_val, max_val = 0, len(items) - 1
        pos = self.integers(min_val, max_val)
        return items[pos]

    def get_all_pairs(self, sameby: Union[str, ColumnList, ColumnDict],
                      diffby: Union[str, ColumnList, ColumnDict]):
        '''
        Get all pairs with given params
        '''
        if isinstance(sameby, dict):
            sameby_all, sameby_any = sameby.get("all", []), sameby.get("any", [])
            if len(sameby_any) == 1:
                raise ValueError('sameby: any should have more than one column')
        else:
            sameby_all = [sameby] if isinstance(sameby, str) else sameby
            sameby_any = []
        
        if isinstance(diffby, dict):
            diffby_all, diffby_any = diffby.get("all", []), diffby.get("any", [])
            if len(diffby_any) == 1:
                raise ValueError('diffby: any should have more than one column')
        else:
            diffby_all = [diffby] if isinstance(diffby, str) else diffby
            diffby_any = []
        
        if set(sameby_all) & set(sameby_any) & set(diffby_all) & set(diffby_any):
            raise ValueError('sameby and diffby must be disjoint lists')
        if not sameby_all and not sameby_any and not diffby_all and not diffby_any:
            raise ValueError('sameby, diffby: at least one column should be provided')
        if not set(sameby_all + sameby_any + diffby_all + diffby_any).issubset(self.columns):
            missing = set(sameby_all + sameby_any + diffby_all + diffby_any) - set(self.columns)
            raise ValueError(f'sameby, diffby: {missing} columns not in DataFrame')
        
        if not sameby_all and not sameby_any:
            if not diffby_any:
                return self._only_diffby_all(diffby_all)
            elif not diffby_all:
                return self._only_diffby_any(diffby_any)
            else:
                return self._only_diffby_all_any(diffby_all, diffby_any)
            
        if sameby_all:
            if len(sameby_all) == 1:
                key = next(iter(sameby_all))
                pairs = self._get_all_pairs_single(key, diffby_all, diffby_any)
            else:
                # TODO: modify to account for:
                #  - and/or sameby_any cols
                ComposedKey = namedtuple('ComposedKey', sameby_all)
                # Multiple sameby. Ordering by minimum number of posible pairs
                sameby_all = sorted(sameby_all, key=self.col_order.get)
                candidates = self._get_all_pairs_single(sameby_all[0], diffby_all, diffby_any)
                col_ix = [self.col_to_ix[col] for col in sameby_all[1:]]

                pairs = dict()
                for key, indices in candidates.items():
                    for id1, id2 in indices:
                        row1 = self.values[id1]
                        row2 = self.values[id2]
                        if np.all(row1[col_ix] == row2[col_ix]):
                            vals = key, *row1[col_ix]
                            key_tuple = ComposedKey(**dict(zip(sameby_all, vals)))
                            pair = (id1, id2)
                            pairs.setdefault(key_tuple, list()).append(pair)
        else:
            pairs = dict()

        if sameby_any:
            if pairs:
                pair_values = np.asarray([v[0] for v in pairs.values()])
                pairs_any = self._filter_pairs_by_condition(pair_values, sameby_any, condition="any_same")
                pairs = {k: v for k, v in pairs.items() if v in pairs_any}
            else:
                pairs = set()
                for col in sameby_any:
                    col_pairs = self._get_all_pairs_single(col, diffby_all, diffby_any)
                    pairs.update({tuple(v[0]) for v in col_pairs.values()})
                pairs = list(pairs)
                pairs.sort(key=lambda x: (x[0], x[1]))
                pairs = {None: pairs}

        
        return pairs

    def _get_all_pairs_single(self, sameby: str, diffby_all: ColumnList, diffby_any: ColumnList):
        '''Get all valid pairs for a single column.'''
        mapper = self.reverse[sameby]
        pairs = dict()
        for key, rows in mapper.items():
            processed = set()
            for id1 in rows:
                valid = set(rows)
                processed.add(id1)
                valid -= processed
                valid = self._filter_diffby(id1, diffby_all, diffby_any, valid)
                for id2 in valid:
                    pair = (id1, id2)
                    pairs.setdefault(key, list()).append(pair)
        return pairs

    def _only_diffby_all(self, diffby_all: ColumnList):
        '''Generate a dict with single NaN key containing all of the pairs
        with different values in the column list'''
        diffby_all = sorted(diffby_all, key=self.col_order.get)

        # Cartesian product for one of the diffby columns
        mapper = self.reverse[diffby_all[0]]
        pairs = self._get_full_pairs(mapper)
        
        if len(diffby_all) > 1:
            pairs = self._filter_pairs_by_condition(pairs, diffby_all[1:], condition="all_diff")
        
        pairs = np.unique(pairs, axis=0)
        return {None: list(map(tuple, pairs))}
    
    def _only_diffby_any(self, diffby: ColumnList):
        '''Generate a dict with single NaN key containing all of the pairs
        with different values in any of specififed columns'''
        diffby = sorted(diffby, key=self.col_order.get)

        pairs = [] 
        for diff_col in diffby:
            mapper = self.reverse[diff_col]
            pairs.extend(self._get_full_pairs(mapper))

        pairs = np.sort(np.asarray(pairs))
        pairs = np.unique(pairs, axis=0)
        return {None: list(map(tuple, pairs))}
    
    def _only_diffby_all_any(self, diffby_all: ColumnList, diffby_any: ColumnList):
        '''Generate a dict with single NaN key containing all of the pairs
        with different values in any of specififed columns'''
        diffby_all_pairs = np.asarray(self._only_diffby_all(diffby_all)[None])
        diffby_all_any = self._filter_pairs_by_condition(diffby_all_pairs, diffby_any, condition="any_diff")
        return {None: list(map(tuple, diffby_all_any))}

    def _filter_diffby(self, idx: int, diffby_all: ColumnList, diffby_any: ColumnList, valid: Set[int]):
        '''
        Remove from valid rows that have matches with idx in any of the diffby columns
        :idx: index of the row to be compared
        :diffby: indices of columns that should have different values
        :valid: candidate rows to be evaluated
        :returns: subset of valid after removing indices

        '''
        row = self.values[idx]
        for col in diffby_all:
            val = row[self.col_to_ix[col]]
            if pd.isna(val):
                continue
            mapper = self.reverse[col]
            valid = valid - mapper[val]
        if diffby_any:
            mapped = []
            for col in diffby_any:
                val = row[self.col_to_ix[col]]
                if pd.isna(val):
                    continue
                mapper = self.reverse[col]
                mapped.append(mapper[val])
            valid = valid - set.intersection(*mapped)
        return valid


    def _get_full_pairs(self, mapper):
        pairs = []
        for key_a, key_b in itertools.combinations(mapper.keys(), 2):
            pairs.extend(itertools.product(mapper[key_a], mapper[key_b]))
        pairs = np.array(pairs)
        return pairs
    

    def _filter_pairs_by_condition(self, pairs, columns, condition="all_same"):
        col_ix = [self.col_to_ix[col] for col in columns]
        vals_a = self.values[pairs[:, 0]][:, col_ix]
        vals_b = self.values[pairs[:, 1]][:, col_ix]

        if "same" in condition:
            valid = vals_a == vals_b
        elif "diff" in condition:
            valid = vals_a != vals_b

        if "all" in condition:
            valid = np.all(valid, axis=1)
        elif "any" in condition:
            valid = np.any(valid, axis=1)

        return pairs[valid]


class MatcherMultilabel():

    def __init__(self, dframe: pd.DataFrame, columns: ColumnList,
                 multilabel_col: str, seed: int):
        self.multilabel_col = multilabel_col
        self.size = dframe.shape[0]
        self.multilabel_set = dframe[multilabel_col].apply(set)
        dframe = dframe.explode(multilabel_col)
        dframe = dframe.reset_index(names='__original_index')
        self.original_index = dframe['__original_index']
        self.matcher = Matcher(dframe, columns, seed)

    def get_all_pairs(self, sameby: Union[str, ColumnList],
                      diffby: ColumnList):
        diffby_multi = self.multilabel_col in diffby
        if diffby_multi:
            # Multilabel in diffby must be 'ALL' instead of 'ANY'
            # Doing this filter afterwards
            diffby = [col for col in diffby if self.multilabel_col != col]
        if not diffby and not sameby and diffby_multi:
            return self._only_diffby_multi()
        pairs = self.matcher.get_all_pairs(sameby, diffby)
        for key, values in pairs.items():
            values = np.asarray(values)
            # Map to original_index
            values[:, 0] = self.original_index[values[:, 0]]
            values[:, 1] = self.original_index[values[:, 1]]

            # Check all of the values in the multilabel_col are different
            if diffby_multi:
                labels_a = self.multilabel_set.iloc[values[:, 0]]
                labels_b = self.multilabel_set.iloc[values[:, 1]]
                valid = [len(a & b) == 0 for a, b in zip(labels_a, labels_b)]
                values = values[valid]
            pairs[key] = list(zip(*values.T))
        return pairs

    def sample_null_pair(self, diffby: ColumnList, n_tries=5):
        null_pair = self.matcher.sample_null_pair(diffby, n_tries)
        id1, id2 = self.original_index[list(null_pair)].values
        return id1, id2

    def get_null_pairs(self, diffby: ColumnList, size: int, n_tries=5):
        null_pairs = []
        for _ in tqdm(range(size)):
            null_pairs.append(self.matcher.sample_null_pair(diffby, n_tries))
        null_pairs = np.array(null_pairs)
        null_pairs[:, 0] = self.original_index[null_pairs[:, 0]].values
        null_pairs[:, 1] = self.original_index[null_pairs[:, 1]].values
        return null_pairs

    def _only_diffby_multi(self):
        '''Special case when it is filter only by the diffby=multilabel_col'''
        pairs = self.get_all_pairs(self.multilabel_col, [])
        pairs = itertools.chain.from_iterable(pairs.values())
        pairs = set(map(frozenset, pairs))
        all_pairs = itertools.combinations(range(self.size), 2)
        filter_fn = lambda x: set(x) not in pairs
        return {None: list(filter(filter_fn, all_pairs))}
