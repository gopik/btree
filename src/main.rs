use std::time::Instant;
use test_env_log::test;

/// **In memory B-Tree implementation** Reference: <http://opendatastructures.org/>
/// The structure of the tree is defined by B which must be >= 2. All the keys are
/// stored in leaf nodes only. Branch nodes are only used for navigation. Each branch node
/// has at most 2*B-1 keys and 2*B children. Each leaf node has 2*B-1 key/values pairs.
///
/// The tree is built bottom up. Initially the tree is empty with root node being a leaf.
/// As the values are inserted into the tree, the leaf fills up and split.
///
/// Split is handled differently for root, branch node and the leaf node. Whenever root splits,
/// a new root (branch node) is created which has just one key and 2 children. Root starts
/// as a leaf node and becomes a branch node on first split.
///
/// Split of leaf node is handled by creating another leaf node which takes half the keys and
/// values of the current leaf node. Across the 2 leaf nodes, the keys and values remain the
/// same as before split.
///
/// In case of branch node split, the number of keys across the 2 nodes is reduced by one and
/// the key is moved up to the parent as the splitting key. If there were 2B keys after an insert,
/// there will be 2B + 1 children. After the split, existing node has B - 1 keys and B children,
/// the new split will have B keys and B + 1 children. The extra key is moved up to the as the key.
/// This is needed to ensure that branch nodes continue to have 1 more child than the number
/// of keys.

mod btree {
    use log::info;
    use serde::Serialize;
    use std::cmp;
    #[derive(Debug, Serialize)]
    pub enum Node<K, V> {
        Branch(BranchNode<K, V>),
        Leaf(LeafNode<K, V>),
    }

    enum BtreeError<K, V> {
        NodeSplit(K, Node<K, V>),
    }

    pub trait NodeKeyType: cmp::Ord + Clone + std::fmt::Display + std::fmt::Debug {}

    impl<T: cmp::Ord + Clone + std::fmt::Display + std::fmt::Debug> NodeKeyType for T {}

    impl<K: NodeKeyType, V> Node<K, V> {
        fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
            match self {
                Self::Branch(ref mut b) => b.insert(key, value),
                Self::Leaf(ref mut l) => l.insert(key, value),
            }
        }
        fn max_key(&self) -> &K {
            match self {
                Self::Branch(ref b) => b.max_key(),
                Self::Leaf(ref l) => l.max_key(),
            }
        }
    }

    #[derive(Debug, Serialize)]
    struct LeafNode<K, V> {
        keys: Vec<K>,
        #[serde(skip)]
        values: Vec<V>,
        #[serde(skip)]
        branch_factor: usize,
    }

    impl<K: NodeKeyType, V> LeafNode<K, V> {
        fn new(branch_factor: usize) -> Self {
            LeafNode {
                keys: vec![],
                values: vec![],
                branch_factor: branch_factor,
            }
        }
        fn max_key(&self) -> &K {
            &self.keys[self.keys.len() - 1]
        }

        fn maybe_split(&mut self) -> Result<(), BtreeError<K, V>> {
            if self.keys.len() == self.branch_factor {
                info!("splitting at keys len = {}", self.keys.len());
                let mut split_node = LeafNode::new(self.branch_factor);
                let mid = self.keys.len() / 2;
                split_node.keys = self.keys.split_off(mid);
                split_node.values = self.values.split_off(mid);

                info!(
                    "after split current={}, split={}",
                    self.keys.len(),
                    split_node.keys.len()
                );
                Err(BtreeError::NodeSplit(
                    self.keys[mid - 1].clone(),
                    Node::Leaf(split_node),
                ))
            } else {
                Ok(())
            }
        }

        fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
            match self.keys.binary_search(&key) {
                Ok(index) => self.values[index] = value,
                Err(index) => {
                    self.keys.insert(index, key);
                    self.values.insert(index, value);
                }
            }
            self.maybe_split()
        }
    }

    #[derive(Debug, Serialize)]
    struct BranchNode<K, V> {
        keys: Vec<K>,
        children: Vec<Node<K, V>>,
        #[serde(skip)]
        branch_factor: usize,
    }

    impl<K: NodeKeyType, V> BranchNode<K, V> {
        fn new(branch_factor: usize) -> Self {
            BranchNode {
                keys: vec![],
                children: vec![],
                branch_factor: branch_factor,
            }
        }

        fn max_key(&self) -> &K {
            &self.keys[self.keys.len() - 1]
        }

        fn maybe_split(&mut self) -> Result<(), BtreeError<K, V>> {
            if self.keys.len() == self.branch_factor {
                let mut split_node = BranchNode::new(self.branch_factor);
                let mid = self.keys.len() / 2;
                split_node.keys = self.keys.split_off(mid + 1);
                split_node.children = self.children.split_off(mid + 1);
                Err(BtreeError::NodeSplit(
                    self.keys.remove(mid),
                    Node::Branch(split_node),
                ))
            } else {
                Ok(())
            }
        }

        fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
            // Each branch node key len is 1 less than children len
            let index_result = self.keys.binary_search(&key);
            let (index, insert_result) = match index_result {
                Ok(index) | Err(index) => (index, self.children[index].insert(key, value)),
            };
            if let Err(BtreeError::NodeSplit(key, split_node)) = insert_result {
                // key is the max key from existing child after the split
                // Case 1: the split child is not the last child.
                //   We replace the key at index with key and insert the split node in child list.
                //   If this reaches the max branching factor, we split the current node.
                // Case 2: the split child is the last child.
                //   We push a new key from the split child. If this reaches the branching factor,
                //   we split the current node.

                if index != self.children.len() - 1 {
                    // case 1
                    self.keys[index] = key;
                    let split_node_key = split_node.max_key();
                    match self.keys.binary_search(split_node_key) {
                        Ok(_) => {
                            panic!(
                                "Unexpected key={} in branch={:?}",
                                split_node_key, self.keys
                            );
                        }
                        Err(split_index) => {
                            self.keys.insert(split_index, split_node_key.clone());
                            self.children.insert(split_index, split_node);
                        }
                    }
                } else {
                    // case 2, set the key as max key of this node and split node as the last child
                    self.keys.push(key);
                    self.children.push(split_node);
                }
            }
            self.maybe_split()
        }
    }

    pub struct BTree<K: cmp::Ord, V> {
        pub root: std::cell::Cell<Node<K, V>>,
        branch_factor: usize, // 2*B - 1
    }

    impl<K: NodeKeyType, V> BTree<K, V> {
        pub fn new(b: usize) -> Self {
            if b < 2 {
                panic!("{}", "B must be at least 2");
            }
            let branch_factor = 2 * b - 1;
            BTree {
                root: std::cell::Cell::new(Node::Leaf(LeafNode::new(branch_factor))),
                branch_factor,
            }
        }
        pub fn insert(&mut self, key: K, value: V) {
            match self.root.get_mut().insert(key, value) {
                Ok(_) => (),
                Err(BtreeError::NodeSplit(key, split_node)) => {
                    let branch_node = BranchNode::<K, V>::new(self.branch_factor);
                    let old_root = self.root.replace(Node::Branch(branch_node));
                    let new_root = self.root.get_mut();
                    if let Node::Branch(ref mut node) = new_root {
                        // Every branch node maintains an invariant that it has m keys and
                        // m+1 children.
                        node.keys.push(key);
                        node.children.push(old_root);
                        node.children.push(split_node);
                    }
                }
            }
        }
    }
}

use btree::*;

#[test]
fn btree_print() {
    let mut btree = BTree::<String, i32>::new(2);
    assert_eq!(
        "Leaf(LeafNode { keys: [], values: [], branch_factor: 3 })",
        format!("{:?}", btree.root.get_mut())
    );
}

#[test]
fn btree_first_node() {
    let mut btree: BTree<String, i32> = BTree::new(2);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);

    assert_eq!(
        "Leaf(LeafNode { keys: [\"one\", \"two\"], values: [1, 2], branch_factor: 3 })",
        format!("{:?}", btree.root.get_mut())
    );
}
fn main() {
    let mut bt = BTree::<i32, i32>::new(128);
    for i in 1..1000_000 {
        bt.insert(i, i);
    }
    let before = Instant::now();
    let mut bt1 = BTree::<i32, i32>::new(128);
    for i in 1..1000_000 {
        bt1.insert(i, i);
    }
    println!("elapsed = {} micros", before.elapsed().as_micros());
}

#[test]
fn btree_leaf_split() {
    let mut btree: BTree<String, i32> = BTree::new(2);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);
    btree.insert(String::from("three"), 3);

    assert_eq!("Branch(BranchNode { keys: [\"one\"], children: [Leaf(LeafNode { keys: [\"one\"], values: [1], branch_factor: 3 }), Leaf(LeafNode { keys: [\"three\", \"two\"], values: [3, 2], branch_factor: 3 })], branch_factor: 3 })", format!("{:?}", btree.root.get_mut()));
}

#[test]
fn btree_branch_split() {
    let mut btree: BTree<String, i32> = BTree::new(3);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);
    btree.insert(String::from("three"), 3);
    btree.insert(String::from("four"), 3);
    btree.insert(String::from("five"), 3);
    btree.insert(String::from("six"), 3);
    btree.insert(String::from("seven"), 3);
    btree.insert(String::from("eight"), 3);
    btree.insert(String::from("nine"), 3);

    println!(
        "{}",
        serde_json::to_string_pretty(btree.root.get_mut())
            .ok()
            .unwrap()
    );

    assert_eq!(
        "{\"Branch\":{\"keys\":[\"four\",\"seven\"],\"children\":[{\"Leaf\":{\"keys\":[\"eight\",\"five\",\"four\"]}},{\"Leaf\":{\"keys\":[\"nine\",\"one\",\"seven\"]}},{\"Leaf\":{\"keys\":[\"six\",\"three\",\"two\"]}}]}}",
        format!(
            "{}",
            serde_json::to_string(btree.root.get_mut()).ok().unwrap()
        )
    )
}

#[test]
fn vec_binary_search() {
    let v = vec![10, 100];
    assert_eq!(Ok(0), v.binary_search(&10));
    assert_eq!(Err(0), v.binary_search(&9));
    assert_eq!(Err(1), v.binary_search(&11));
    assert_eq!(Ok(1), v.binary_search(&100));
    assert_eq!(Err(2), v.binary_search(&200));
}
