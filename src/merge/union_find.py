"""Union-Find module"""

# pylint: disable=invalid-name

from collections import defaultdict

import numpy as np
import numpy.typing as npt


class UnionFind:
    """Union-Find data structure.

    Attributes
    ----------
    n : `int`
        Number of nodes.
    parents : `npt.NDArray[np.int32]`
        Two different meaning depending on the sign. If `parents[i] < 0`, it means the number
        of descendants including the node `i` multiplied by -1. If `parents[i] >= 0`, it indicates
        the index of `i`'s parent.
    member_set : list[set[int]]
        Set of member indices of the group each node belongs to.
    """

    def __init__(self, n: int, label_indices: npt.NDArray[np.int32]):
        """Initializes the Union-Find data structure.

        Parameters
        ----------
        n : `int`
            Number of nodes.
        label_indices : `npt.NDArray[np.int32]`
            Label indices (IDs) of each point. The label ID of point `i` is label_indices[i].
        """
        self.n = n
        self.parents: npt.NDArray[np.int32] = np.ones((n), dtype=np.int32) * -1
        self.member_set: list[set[int]] = [{label_indices[i]} for i in range(n)]

    def find(self, x: int) -> int:
        """Finds the parent of the given node.

        Parameters
        ----------
        x : `int`
            Index of the node whose parent to find

        Returns
        -------
        parent : `int`
            Index of the parent
        """
        # If the parent index is negative,
        if self.parents[x] < 0:
            # it means the node `x` itself is a parent
            return x
        # Otherwise, recursively search the parent
        root: int = self.find(self.parents[x])
        # and update the parent information (PATH COMPRESSION)
        self.parents[x] = root

        return self.parents[x]

    def union(self, x: int, y: int) -> None:
        """Merges two groups of the given two nodes (Union by Size).

        Parameters
        ----------
        x, y : `int`
            Node that should be merged
        """

        # Finds the parent of each
        x_: int = self.find(x)
        y_: int = self.find(y)

        # If both parents are the same,
        if x_ == y_:
            # it means they belong to the same group
            return

        # Set `x` to be the one with more descendants
        if self.parents[x_] > self.parents[y_]:
            x_, y_ = y_, x_

        # Node with more descendants (`x`) becomes the parent
        self.parents[x_] += self.parents[y_]
        self.parents[y_] = x_

        self.member_set[x_] |= self.member_set[y_]

    def members(self, x: int) -> set[int]:
        """Returns the member list the given node belongs to.

        Parameters
        ----------
        x : `int`
            Index of the node whose group members to return

        Returns
        -------
        members : `set[int]`
            Index of the group members
        """

        root: int = self.find(x)
        members: set[int] = self.member_set[root]

        return members

    def all_group_members(self) -> dict[int, list[int]]:
        """
        Lists up all members of each group.

        Returns
        -------
        group_members : dict[int, List[int]]
            Dictionary whose keys are the root of each group and whose values are the
            lists of the members.
        """

        group_members: defaultdict[int, list[int]] = defaultdict(list)
        for member in range(self.n):
            root: int = self.find(member)
            group_members[root].append(member)

        return group_members

    def __str__(self) -> str:
        """Returns a formatted string for `print`.

        Returns
        -------
        formatted_string : `str`
            String object shown when called by `print`
        """

        formatted_string = "\n".join(
            f"{root}: {members}" for root, members in self.all_group_members().items()
        )
        return formatted_string


if __name__ == "__main__":
    n = 6
    label_indices: npt.NDArray[np.int32] = np.arange(n)
    union_find = UnionFind(n, label_indices)
    union_find.union(0, 1)
    print(union_find)
