use log::info;
use std::cmp;

use serde::Serialize;

/// In memory btree implementation
/// Btree has 2 types of nodes: Leaf nodes that contain values and branch nodes that branch to leaf or branch nodes based on keys.

#[derive(Debug, Serialize)]
enum Node<K, V> {
    Branch(BranchNode<K, V>),
    Leaf(LeafNode<K, V>),
}

enum BtreeError<K, V> {
    NodeSplit(K, Node<K, V>),
}

trait NodeKeyType: cmp::Ord + Clone + std::fmt::Display + std::fmt::Debug {}

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
    fn insert(&mut self, key: K, value: V) -> Result<(), BtreeError<K, V>> {
        match self.keys.binary_search(&key) {
            Ok(index) => self.values[index] = value,
            Err(index) => {
                self.keys.insert(index, key);
                self.values.insert(index, value);
            }
        }

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
}

struct BTree<K: cmp::Ord, V> {
    root: std::cell::Cell<Node<K, V>>,
    branch_factor: usize, // 2*B
}

#[cfg(test)]
impl<K: NodeKeyType, V> BTree<K, V> {
    fn new(B: usize) -> Self {
        if B < 2 {
            panic!("{}", "B must be at least 2");
        }
        let branch_factor = 2 * B - 1;
        BTree {
            root: std::cell::Cell::new(Node::Leaf(LeafNode::new(branch_factor))),
            branch_factor,
        }
    }
    fn insert(&mut self, key: K, value: V) {
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
    println!("Hello, world!");
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

#[test]
fn vec_init_with_length() {
    let y: usize = 8;
    let v = vec![-1; y];
}
