#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace resple2 {

template <class T, std::size_t BlockSize = 256>
class ReusablePointBuffer {
    static_assert(BlockSize > 0);
    using Slot = std::aligned_storage_t<sizeof(T), alignof(T)>;
    struct Block { Slot slots[BlockSize]; };
    std::vector<std::unique_ptr<Block>> owned_;
    std::vector<Block*> free_;
    std::deque<Block*> active_;
    std::size_t front_ = 0, size_ = 0;

    static T* object(Block* block, std::size_t offset) {
        return std::launder(reinterpret_cast<T*>(&block->slots[offset]));
    }
    void appendBlock() {
        if (free_.empty()) {
            if (free_.capacity() < owned_.size() + 1)
                free_.reserve(std::max(owned_.size() + 1, free_.capacity() * 2));
            std::unique_ptr<Block> block(new Block);
            owned_.push_back(std::move(block));
            free_.push_back(owned_.back().get());
        }
        active_.push_back(free_.back());
        free_.pop_back();
    }

public:
    using value_type = T;
    template <bool Const> class Iterator {
        friend class ReusablePointBuffer;
        template <bool> friend class Iterator;
        using Buffer = std::conditional_t<Const, const ReusablePointBuffer, ReusablePointBuffer>;
        Buffer* buffer_ = nullptr;
        std::size_t index_ = 0;
        Iterator(Buffer* buffer, std::size_t index) : buffer_(buffer), index_(index) {}
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = std::conditional_t<Const, const T*, T*>;
        using reference = std::conditional_t<Const, const T&, T&>;
        Iterator() = default;
        template <bool C = Const, std::enable_if_t<C, int> = 0>
        Iterator(const Iterator<false>& other) : buffer_(other.buffer_), index_(other.index_) {}
        reference operator*() const { return (*buffer_)[index_]; }
        pointer operator->() const { return std::addressof(**this); }
        Iterator& operator++() { ++index_; return *this; }
        Iterator operator++(int) { auto previous = *this; ++*this; return previous; }
        template <bool C> bool operator==(const Iterator<C>& other) const {
            return buffer_ == other.buffer_ && index_ == other.index_;
        }
        template <bool C> bool operator!=(const Iterator<C>& other) const { return !(*this == other); }
    };
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    ReusablePointBuffer() = default;
    ReusablePointBuffer(const ReusablePointBuffer&) = delete;
    ReusablePointBuffer& operator=(const ReusablePointBuffer&) = delete;
    ~ReusablePointBuffer() { clear(); }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    std::size_t allocatedBlocks() const { return owned_.size(); }
    std::size_t retainedBytes() const { return owned_.size() * sizeof(Block); }

    template <class... Args> T& emplace_back(Args&&... args) {
        const std::size_t index = front_ + size_;
        const bool new_block = index % BlockSize == 0;
        if (new_block) appendBlock();
        T* item;
        try {
            item = ::new (static_cast<void*>(&active_.back()->slots[index % BlockSize]))
                T(std::forward<Args>(args)...);
        } catch (...) {
            if (new_block) { free_.push_back(active_.back()); active_.pop_back(); }
            throw;
        }
        ++size_;
        return *item;
    }
    T& operator[](std::size_t i) {
        return *object(active_[(front_ + i) / BlockSize], (front_ + i) % BlockSize);
    }
    const T& operator[](std::size_t i) const {
        return *object(active_[(front_ + i) / BlockSize], (front_ + i) % BlockSize);
    }
    T& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("point buffer index");
        return (*this)[i];
    }
    const T& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("point buffer index");
        return (*this)[i];
    }
    T& front() { return (*this)[0]; }
    const T& front() const { return (*this)[0]; }
    T& back() { return (*this)[size_ - 1]; }
    const T& back() const { return (*this)[size_ - 1]; }
    void pop_front() { pop_front(1); }
    void pop_front(std::size_t count) {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t i = 0; i < count; ++i) (*this)[i].~T();
        }
        size_ -= count;
        front_ += count;
        while (front_ >= BlockSize) {
            free_.push_back(active_.front());
            active_.pop_front();
            front_ -= BlockSize;
        }
        if (size_ == 0 && front_ != 0) {
            free_.push_back(active_.front());
            active_.pop_front();
            front_ = 0;
        }
    }
    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t i = 0; i < size_; ++i) (*this)[i].~T();
        }
        for (auto* block : active_) free_.push_back(block);
        active_.clear();
        front_ = size_ = 0;
    }
};

template <class T, std::size_t BlockSize = 256>
class RetainedPointBuffer {
    static_assert(BlockSize > 0);
    using Slot = std::aligned_storage_t<sizeof(T), alignof(T)>;
    struct Block {
        Slot slots[BlockSize];
        std::size_t constructed = 0, owners = 0;
    };
    static T* object(Block* block, std::size_t offset) {
        return std::launder(reinterpret_cast<T*>(&block->slots[offset]));
    }
    struct Pool {
        std::vector<std::unique_ptr<Block>> owned;
        std::vector<Block*> free;
        Block* acquire() {
            if (free.empty()) {
                if (free.capacity() < owned.size() + 1)
                    free.reserve(std::max(owned.size() + 1, free.capacity() * 2));
                std::unique_ptr<Block> block(new Block);
                owned.push_back(std::move(block));
                free.push_back(owned.back().get());
            }
            Block* block = free.back();
            free.pop_back();
            block->owners = 1;
            return block;
        }
        void release(Block* block) noexcept {
            if (--block->owners) return;
            if constexpr (!std::is_trivially_destructible_v<T>)
                for (std::size_t i = 0; i < block->constructed; ++i) object(block, i)->~T();
            block->constructed = 0;
            free.push_back(block);
        }
    };
    std::shared_ptr<Pool> pool_ = std::make_shared<Pool>();
    std::deque<Block*> active_;
    std::size_t front_ = 0, size_ = 0;
    void appendBlock() {
        Block* block = pool_->acquire();
        try { active_.push_back(block); }
        catch (...) { pool_->release(block); throw; }
    }

public:
    class Lease {
        friend class RetainedPointBuffer;
        std::shared_ptr<Pool> pool_;
        Block* block_ = nullptr;
        std::size_t start_ = 0, size_ = 0;
        Lease(std::shared_ptr<Pool> pool, Block* block, std::size_t start, std::size_t size)
            : pool_(std::move(pool)), block_(block), start_(start), size_(size) { ++block_->owners; }
    public:
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept
            : pool_(std::move(other.pool_)), block_(std::exchange(other.block_, nullptr)),
              start_(other.start_), size_(other.size_) {}
        Lease& operator=(Lease&&) = delete;
        ~Lease() { if (block_) pool_->release(block_); }
        std::size_t size() const { return size_; }
        const T& operator[](std::size_t i) const {
            return *object(block_, start_ + i);
        }
    };
    Lease pin_front(std::size_t count) const {
        return Lease(pool_, active_.front(), front_, std::min(count, BlockSize - front_));
    }
    using value_type = T;
    template <bool Const> class Iterator {
        friend class RetainedPointBuffer;
        template <bool> friend class Iterator;
        using Buffer = std::conditional_t<Const, const RetainedPointBuffer, RetainedPointBuffer>;
        Buffer* buffer_ = nullptr;
        std::size_t index_ = 0;
        Iterator(Buffer* buffer, std::size_t index) : buffer_(buffer), index_(index) {}
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = std::conditional_t<Const, const T*, T*>;
        using reference = std::conditional_t<Const, const T&, T&>;
        Iterator() = default;
        template <bool C = Const, std::enable_if_t<C, int> = 0>
        Iterator(const Iterator<false>& other) : buffer_(other.buffer_), index_(other.index_) {}
        reference operator*() const { return (*buffer_)[index_]; }
        pointer operator->() const { return std::addressof(**this); }
        Iterator& operator++() { ++index_; return *this; }
        Iterator operator++(int) { auto previous = *this; ++*this; return previous; }
        template <bool C> bool operator==(const Iterator<C>& other) const {
            return buffer_ == other.buffer_ && index_ == other.index_;
        }
        template <bool C> bool operator!=(const Iterator<C>& other) const { return !(*this == other); }
    };
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;
    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size_); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size_); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }
    RetainedPointBuffer() = default;
    RetainedPointBuffer(const RetainedPointBuffer&) = delete;
    RetainedPointBuffer& operator=(const RetainedPointBuffer&) = delete;
    ~RetainedPointBuffer() { clear(); }

    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    std::size_t allocatedBlocks() const { return pool_->owned.size(); }
    std::size_t retainedBytes() const { return pool_->owned.size() * sizeof(Block); }

    template <class... Args> T& emplace_back(Args&&... args) {
        const std::size_t index = front_ + size_;
        const bool new_block = index % BlockSize == 0;
        if (new_block) appendBlock();
        T* item;
        try {
            item = ::new (static_cast<void*>(&active_.back()->slots[index % BlockSize]))
                T(std::forward<Args>(args)...);
        } catch (...) {
            if (new_block) { pool_->release(active_.back()); active_.pop_back(); }
            throw;
        }
        ++active_.back()->constructed;
        ++size_;
        return *item;
    }
    T& operator[](std::size_t i) {
        return *object(active_[(front_ + i) / BlockSize], (front_ + i) % BlockSize);
    }
    const T& operator[](std::size_t i) const {
        return *object(active_[(front_ + i) / BlockSize], (front_ + i) % BlockSize);
    }
    T& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("point buffer index");
        return (*this)[i];
    }
    const T& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("point buffer index");
        return (*this)[i];
    }
    T& front() { return (*this)[0]; }
    const T& front() const { return (*this)[0]; }
    T& back() { return (*this)[size_ - 1]; }
    const T& back() const { return (*this)[size_ - 1]; }
    void pop_front() { pop_front(1); }
    void pop_front(std::size_t count) {
        size_ -= count;
        front_ += count;
        while (front_ >= BlockSize) {
            pool_->release(active_.front());
            active_.pop_front();
            front_ -= BlockSize;
        }
        if (size_ == 0 && front_ != 0) {
            pool_->release(active_.front());
            active_.pop_front();
            front_ = 0;
        }
    }
    void clear() noexcept {
        for (auto* block : active_) pool_->release(block);
        active_.clear();
        front_ = size_ = 0;
    }
};

template <class Input, class Workspace, std::size_t BlockSize = 256>
class ActivePointBuffer {
    using Pending = RetainedPointBuffer<Input, BlockSize>;
    struct Segment {
        typename Pending::Lease lease;
        std::size_t remaining = 0;
        explicit Segment(typename Pending::Lease&& source) : lease(std::move(source)) {}
    };
    std::deque<Segment> segments_;
    ReusablePointBuffer<Workspace, BlockSize> points_;
public:
    using value_type = Workspace;
    ActivePointBuffer() = default;
    ActivePointBuffer(const ActivePointBuffer&) = delete;
    ActivePointBuffer& operator=(const ActivePointBuffer&) = delete;
    ~ActivePointBuffer() { clear(); }

    void append_from(Pending& source, std::size_t count) {
        while (count) {
            segments_.emplace_back(source.pin_front(count));
            auto& segment = segments_.back();
            try {
                for (std::size_t i = 0; i < segment.lease.size(); ++i) {
                    points_.emplace_back(segment.lease[i]);
                    ++segment.remaining;
                    source.pop_front();
                    --count;
                }
            } catch (...) {
                if (!segment.remaining) segments_.pop_back();
                throw;
            }
        }
    }
    void pop_front(std::size_t count = 1) {
        points_.pop_front(count);
        while (count) {
            auto& segment = segments_.front();
            const auto consumed = std::min(count, segment.remaining);
            segment.remaining -= consumed;
            count -= consumed;
            if (!segment.remaining) segments_.pop_front();
        }
    }
    void clear() noexcept { points_.clear(); segments_.clear(); }
    std::size_t size() const { return points_.size(); }
    bool empty() const { return points_.empty(); }
    std::size_t allocatedBlocks() const { return points_.allocatedBlocks(); }
    std::size_t retainedBytes() const { return points_.retainedBytes(); }
    Workspace& operator[](std::size_t i) { return points_[i]; }
    const Workspace& operator[](std::size_t i) const { return points_[i]; }
    Workspace& at(std::size_t i) { return points_.at(i); }
    const Workspace& at(std::size_t i) const { return points_.at(i); }
    Workspace& front() { return points_.front(); }
    const Workspace& front() const { return points_.front(); }
    Workspace& back() { return points_.back(); }
    const Workspace& back() const { return points_.back(); }
    auto begin() { return points_.begin(); }
    auto end() { return points_.end(); }
    auto begin() const { return points_.begin(); }
    auto end() const { return points_.end(); }
    auto cbegin() const { return points_.cbegin(); }
    auto cend() const { return points_.cend(); }
};

}
