use std::cmp;
use std::fmt;

use serde::{Deserialize, Serialize};

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
        if self.keys.len() < self.branch_factor {
            let index_result = self.keys.binary_search(&key);
            let index;
            match index_result {
                Ok(i) => index = i,
                Err(i) => {
                    index = i;
                    self.keys.insert(i, key);
                }
            }
            self.values.insert(index, value);
            Ok(())
        } else {
            // TODO(gopik): Only split when the key to be inserted is not already
            // in the node.

            // Split current node into 2 by moving half of the entries
            // to the new node. Push the new key/val into appropriate node.
            let mut split_node = LeafNode::new(self.branch_factor);
            let mid = self.keys.len() / 2;
            split_node.keys = self.keys.split_off(mid);
            split_node.values = self.values.split_off(mid);

            let self_key = self.keys[mid - 1].clone();
            let insert_node = if key < self.keys[mid - 1] {
                self
            } else {
                &mut split_node
            };

            match insert_node.keys.binary_search(&key) {
                Ok(index) => {
                    insert_node.values[index] = value;
                }
                Err(index) => {
                    insert_node.keys.insert(index, key);
                    insert_node.values.insert(index, value);
                }
            }

            // TODO(gopik): Should split be returned in error or ok?
            Err(BtreeError::NodeSplit(self_key, Node::Leaf(split_node)))
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
        let index_result = self.keys.binary_search(&key);
        let (index, insert_result) = match index_result {
            Ok(index) => {
                // self.keys[index] == key
                (index, self.children[index].insert(key, value))
            }
            Err(index) => {
                // self.keys[index] > key
                if index < self.keys.len() {
                    (index, self.children[index].insert(key, value))
                } else {
                    self.keys[index - 1] = key.clone();
                    (index - 1, self.children[index - 1].insert(key, value))
                }
            }
        };
        if let Err(BtreeError::NodeSplit(key, split_node)) = insert_result {
            self.keys[index] = key;
            let split_key = split_node.max_key();
            let split_key_index_result = self.keys.binary_search(split_key);

            match split_key_index_result {
                Ok(split_key_index) => {
                    panic!("Unexpected key={} in branch={:?}", split_key, self.keys);
                }
                Err(split_key_index)
                    if split_key_index < self.keys.len()
                        && self.keys.len() < self.branch_factor =>
                {
                    self.keys.insert(split_key_index, split_key.clone());
                    self.children.insert(split_key_index, split_node);
                    Ok(())
                }
                Err(split_key_index) if self.keys.len() < self.branch_factor => {
                    self.keys.push(split_key.clone());
                    self.children.push(split_node);
                    Ok(())
                }
                _ => {
                    // Branch node needs to split.
                    let mut split_branch_node = BranchNode::new(self.branch_factor);
                    let mid = self.keys.len() / 2;
                    split_branch_node.keys = self.keys.split_off(mid);
                    split_branch_node.children = self.children.split_off(mid);

                    let self_key = self.keys[mid - 1].clone();
                    let insert_node = if split_key <= &self_key {
                        self
                    } else {
                        &mut split_branch_node
                    };

                    match insert_node.keys.binary_search(split_key) {
                        Ok(idx) => insert_node.children[idx] = split_node,
                        Err(idx) => {
                            insert_node.keys.insert(idx, split_key.clone());
                            insert_node.children.insert(idx, split_node);
                        }
                    }

                    // TODO(gopik): Should split be returned in error or ok?
                    Err(BtreeError::NodeSplit(
                        self_key,
                        Node::Branch(split_branch_node),
                    ))
                }
            }
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
    fn new(branch_factor: usize) -> Self {
        BTree {
            root: std::cell::Cell::new(Node::Leaf(LeafNode::new(branch_factor))),
            branch_factor: branch_factor,
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
                    node.keys.push(key);
                    node.children.push(old_root);
                    node.keys.push(split_node.max_key().clone());
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
        "Leaf(LeafNode { keys: [], values: [], branch_factor: 2 })",
        format!("{:?}", btree.root.get_mut())
    );
}

#[test]
fn btree_first_node() {
    let mut btree: BTree<String, i32> = BTree::new(2);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);

    assert_eq!(
        "Leaf(LeafNode { keys: [\"one\", \"two\"], values: [1, 2], branch_factor: 2 })",
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

    assert_eq!("Branch(BranchNode { keys: [\"one\", \"two\"], children: [Leaf(LeafNode { keys: [\"one\"], values: [1], branch_factor: 2 }), Leaf(LeafNode { keys: [\"three\", \"two\"], values: [3, 2], branch_factor: 2 })], branch_factor: 2 })", format!("{:?}", btree.root.get_mut()));
}

#[test]
fn btree_branch_split() {
    let mut btree: BTree<String, i32> = BTree::new(2);
    btree.insert(String::from("one"), 1);
    btree.insert(String::from("two"), 2);
    btree.insert(String::from("three"), 3);
    btree.insert(String::from("four"), 3);
    btree.insert(String::from("five"), 3);

    println!(
        "{}",
        serde_json::to_string_pretty(btree.root.get_mut())
            .ok()
            .unwrap()
    );

    assert_eq!(
        "",
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
