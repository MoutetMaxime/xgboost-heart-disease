class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


class Tree:
    def __init__(self):
        self.root = None

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
        else:
            self._insert(self.root, data)

    def _insert(self, node, data):
        if data < node.data:
            if node.left is None:
                node.left = Node(data)
            else:
                self._insert(node.left, data)
        elif data > node.data:
            if node.right is None:
                node.right = Node(data)
            else:
                self._insert(node.right, data)

    def inorder(self):
        self._inorder(self.root)
        print()

    def _inorder(self, node):
        if node is not None:
            self._inorder(node.left)
            print(node.data, end=" ")
            self._inorder(node.right)

    def preorder(self):
        self._preorder(self.root)
        print()

    def _preorder(self, node):
        if node is not None:
            print(node.data, end=" ")
            self._preorder(node.left)
            self._preorder(node.right)

    def postorder(self):
        self._postorder(self.root)
        print()

    def _postorder(self, node):
        if node is not None:
            self._postorder(node.left)
            self._postorder(node.right)
            print(node.data, end=" ")
