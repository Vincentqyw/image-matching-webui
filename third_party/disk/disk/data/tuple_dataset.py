class TupleDataset:
    def __init__(self, item_dataset, tuples):
        self.item_dataset = item_dataset
        self.tuples       = tuples

    def tuple_transform(self, tuple_):
        return tuple_

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        items = tuple(self.item_dataset[i] for i in self.tuples[idx]) 

        return self.tuple_transform(items)
