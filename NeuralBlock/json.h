#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>

namespace NNLib {

    class JSON {

    private:
        union Value {
            bool m_bool;
            int m_int;
            double m_double;
            std::string* m_string;
            std::vector<JSON>* m_array;
            std::map<std::string, JSON>* m_object;
        };

    public:
        enum Type {
            json_null = 0,
            json_bool,
            json_int,
            json_double,
            json_string,
            json_array,
            json_object
        };

    private:
        Type m_type;
        Value m_value;

    public:
        JSON() : m_type(json_null) {};

        JSON(bool value) : m_type(json_bool) { m_value.m_bool = value; };

        JSON(int value) : m_type(json_int) { m_value.m_int = value; };

        JSON(double value) : m_type(json_double) { m_value.m_double = value; };

        JSON(const char* value) : m_type(json_string) { m_value.m_string = new std::string(value); };

        JSON(const std::string& value) : m_type(json_string) { m_value.m_string = new std::string(value); };

        JSON(Type type) : m_type(type) {
            if (m_type == json_bool) { m_value.m_bool = false; }
            else if (m_type == json_int) { m_value.m_int = 0; }
            else if (m_type == json_double) { m_value.m_double = 0.0; }
            else if (m_type == json_string) { m_value.m_string = new std::string(""); }
            else if (m_type == json_array) { m_value.m_array = new std::vector<JSON>(); }
            else if (m_type == json_object) { m_value.m_object = new std::map<std::string, JSON>(); }
            else {};
        };

        JSON(const JSON& other) {
            // copy the items from other
            copy(other);
        };

        // allow to assign the item to a variable
        operator bool() {
            if (m_type != json_bool) { throw std::invalid_argument("Type Error, not a json_bool type."); };
            return m_value.m_bool;
        };

        operator int() {
            if (m_type != json_int) { throw std::invalid_argument("Type Error, not a json_int type."); };
            return m_value.m_int;
        };

        operator double() {
            if (m_type != json_double) { throw std::invalid_argument("Type Error, not a json_double type."); };
            return m_value.m_double;
        };

        operator std::string() {
            if (m_type != json_string) { throw std::invalid_argument("Type Error, not a json_string type."); };
            return *(m_value.m_string);
        };

        // access array with index
        JSON& operator [] (int idx) {
            if (m_type != json_array) {
                m_type = json_array;
                m_value.m_array = new std::vector<JSON>();
            };
            int size = (m_value.m_array)->size();
            if (idx < 0 || idx > size - 1) { throw std::invalid_argument("Index is out of range"); };
            return (m_value.m_array)->at(idx);
        };

        // delete some types because some of them we use a pointer and new
        void clear() {
            if (m_type == json_bool) { m_value.m_bool = false; }
            else if (m_type == json_int) { m_value.m_int = 0; }
            else if (m_type == json_double) { m_value.m_double = 0.0; }
            else if (m_type == json_string) { delete m_value.m_string; }
            else if (m_type == json_array) {
                for (auto item = (m_value.m_array)->begin(); item != (m_value.m_array)->end(); item++) {
                    item->clear();
                };
                delete m_value.m_array;
            }
            else if (m_type == json_object) {
                for (auto item = (m_value.m_object)->begin(); item != (m_value.m_object)->end(); item++) {
                    (item->second).clear();
                };
                delete m_value.m_object;
            }
            else {};
            m_type = json_null;
        };

        // add item into array
        void append(const JSON& other) {
            if (m_type != json_array) {
                // we perform clean because we need to clear out everything before initializing a new vector
                clear();
                m_type = json_array;
                m_value.m_array = new std::vector<JSON>();
            };
            (m_value.m_array)->push_back(other);
        };

        // print array items out
        std::string str() const {
            std::stringstream ss;
            if (m_type == json_null) { ss << "null"; }
            else if (m_type == json_bool) {
                if (m_value.m_bool) { ss << "true"; }
                else { ss << "false"; };
            }
            else if (m_type == json_int) { ss << m_value.m_int; }
            else if (m_type == json_double) { ss << m_value.m_double; }
            else if (m_type == json_string) { ss << '\"' << *(m_value.m_string) << '\"'; }
            else if (m_type == json_array) {
                ss << "[";
                for (auto item = (m_value.m_array)->begin(); item != (m_value.m_array)->end(); item++) {
                    if (item != (m_value.m_array)->begin()) { ss << ", "; };
                    ss << item->str();
                };
                ss << "]";
            }
            else if (m_type == json_object) {
                ss << "{";
                for (auto item = (m_value.m_object)->begin(); item != (m_value.m_object)->end(); item++) {
                    if (item != (m_value.m_object)->begin()) { ss << ", "; };
                    ss << '\"' << item->first << '\"' << ": " << (item->second).str();
                };
                ss << "}";
            }
            return ss.str();
        };

        // access object key-value
        JSON& operator[] (const char* key) {
            std::string name(key);
            // use the next operator[] function to implement
            return (*(this))[name];
        };

        JSON& operator[] (const std::string& key) {
            if (m_type != json_object) {
                // we perform clean because we need to clear out everything before initializing a new vector
                clear();
                m_type = json_object;
                m_value.m_object = new std::map<std::string, JSON>();
            };
            return (*(m_value.m_object))[key];
        };

        // copy from other object to this one
        void operator = (const JSON& other) {
            clear();
            copy(other);
        };

        // compare two items
        bool operator == (const JSON& other) {
            if (m_type != other.m_type) { return false; };

            if (m_type == json_null) { return true; }
            else if (m_type == json_bool) { return m_value.m_bool == other.m_value.m_bool; }
            else if (m_type == json_int) { return m_value.m_int == other.m_value.m_int; }
            else if (m_type == json_double) { return m_value.m_double == other.m_value.m_double; }
            else if (m_type == json_string) { return *(m_value.m_string) == *(other.m_value.m_string); }
            else if (m_type == json_array) { return m_value.m_array == other.m_value.m_array; }
            else if (m_type == json_object) { return m_value.m_object == other.m_value.m_object; }
            else { return false; };
        };

        bool operator != (const JSON& other) {
            return !((*this) == other);
        };

        void copy(const JSON& other) {
            m_type = other.m_type;
            if (m_type == json_bool) { m_value.m_bool = other.m_value.m_bool; }
            else if (m_type == json_int) { m_value.m_int = other.m_value.m_int; }
            else if (m_type == json_double) { m_value.m_double = other.m_value.m_double; }
            else if (m_type == json_string) { m_value.m_string = other.m_value.m_string; }
            else if (m_type == json_array) { m_value.m_array = other.m_value.m_array; }
            else if (m_type == json_object) { m_value.m_object = other.m_value.m_object; }
            else {};
        };

        typedef std::vector<JSON>::iterator iterator;
        iterator begin() { return (m_value.m_array)->begin(); };
        iterator end() { return (m_value.m_array)->end(); };

        bool is_null() const { return m_type == json_null; };
        bool is_bool() const { return m_type == json_bool; };
        bool is_int() const { return m_type == json_int; };
        bool is_double() const { return m_type == json_double; };
        bool is_string() const { return m_type == json_string; };
        bool is_array() const { return m_type == json_array; };
        bool is_object() const { return m_type == json_object; };

        // bool has(const char * key) {
        //   std::string name(key);
        //   return has(name);
        // };

        bool has(JSON& elem) {
            if (m_type == json_array) {
                for (auto item = (m_value.m_array)->begin(); item != (m_value.m_array)->end(); item++) {
                    if (*(item) == elem) { return true; };
                };
                return false;
            };
            if (m_type == json_object) {
                if (elem.m_type != json_string) { return false; };
                return ((m_value.m_object)->find(elem) != (m_value.m_object)->end());
            };
            return false;
        };

        void remove(JSON& elem) {
            if (m_type == json_array) {
                for (int i = 0; i < (m_value.m_array)->size(); i++) {
                    if (*((m_value.m_array)->begin() + i) == elem) {
                        (m_value.m_array)->at(i).clear();
                        (m_value.m_array)->erase(((m_value.m_array)->begin() + i));
                    };
                };
            };
            if (m_type == json_object) {
                auto item = (m_value.m_object)->find(elem);
                if (item == (m_value.m_object)->end()) { return; };
                (*(m_value.m_object))[elem].clear();
                (m_value.m_object)->erase(elem);
            };
        };
    };


    class Parser {
    private:
        std::string m_str;
        int m_idx;

        void skip_white_space() {
            while (m_str[m_idx] == ' ' || m_str[m_idx] == '\n' || m_str[m_idx] == '\r' || m_str[m_idx] == '\t') {
                m_idx++;
            };
        };

        char get_next_token() {
            skip_white_space();
            return m_str[m_idx++];
        };

        JSON parse_null() {
            if (m_str == "null") {
                m_idx += 4;
                return JSON();
            };
            throw std::invalid_argument("parse_null error");
        };

        JSON parse_bool() {
            if (m_str == "true") {
                m_idx += 4;
                return JSON(true);
            }
            else if (m_str == "false") {
                m_idx += 5;
                return JSON(false);
            };
            throw std::invalid_argument("parse_bool error");
        };

        JSON parse_number() {
            int idx = m_idx;
            if (m_str[m_idx] == '-') { m_idx++; };
            if (m_str[m_idx] < '0' || m_str[m_idx] > '9') { throw std::invalid_argument("parse_number error"); };
            while (m_str[m_idx] >= '0' && m_str[m_idx] <= '9') { m_idx++; };
            if (m_str[m_idx] != '.') {
                int i = std::atoi(m_str.c_str() + idx);
                return JSON(i);
            };
            m_idx++;
            if (m_str[m_idx] < '0' || m_str[m_idx] > '9') { throw std::invalid_argument("parse_number error"); };
            while (m_str[m_idx] >= '0' && m_str[m_idx] <= '9') { m_idx++; };
            double f = std::atof(m_str.c_str() + idx);
            return JSON(f);
        };

        std::string parse_string() {
            std::string out;
            while (true) {
                char ch = m_str[m_idx++];
                if (ch == '"') { break; };
                if (ch == '\\') {
                    ch = m_str[m_idx++];
                    if (ch == '\n') { out += '\n'; }
                    else if (ch == '\r') { out += '\r'; }
                    else if (ch == '\t') { out += '\t'; }
                    else if (ch == '\b') { out += '\b'; }
                    else if (ch == '\f') { out += '\f'; }
                    else if (ch == '"') { out += "\\\""; }
                    else if (ch == '\\') { out += "\\\\"; }
                    else if (ch == 'u') {
                        out += "\\u";
                        for (int i = 0; i < 4; i++) { out += m_str[m_idx++]; };
                    };
                }
                else { out += ch; };
            };
            return out;
        };

        JSON parse_array() {
            JSON arr(JSON::json_array);
            char ch = get_next_token();
            if (ch == ']') { return arr; };
            m_idx--;
            while (true) {
                arr.append(parse());
                ch = get_next_token();
                // if (ch == '\\') { m_idx --; };
                if (ch == ']') { break; };
                if (ch != ',') { throw std::invalid_argument("parse_array error"); };
                m_idx++;
            };
            return arr;
        };

        JSON parse_object() {
            JSON obj(JSON::json_object);
            char ch = get_next_token();
            if (ch == '}') { return obj; };
            m_idx--;
            while (true) {
                ch = get_next_token();
                if (ch != '"') { throw std::invalid_argument("parse_object error"); };
                std::string key = parse_string();
                ch = get_next_token();
                if (ch != ':') { throw std::invalid_argument("parse_object error"); };
                obj[key] = parse();
                ch = get_next_token();
                if (ch == '}') { break; };
                if (ch != ',') { throw std::invalid_argument("parse_object error"); };
                m_idx++;
            };
            return obj;
        };


    public:
        Parser() : m_str(""), m_idx(0) {};

        void load(const std::string& str) {
            m_str = str;
            m_idx = 0;
        };

        JSON parse() {
            char ch = get_next_token();
            if (ch == 'n') {
                m_idx--;
                return parse_null();
            }
            else if (ch == 't' || ch == 'f') {
                m_idx--;
                return parse_bool();
            }
            else if (ch == '-' || ch == '0' || ch == '1' || ch == '2' || ch == '3' || ch == '4' ||
                ch == '5' || ch == '6' || ch == '7' || ch == '8' || ch == '9') {
                m_idx--;
                return parse_number();
            }
            else if (ch == '"') { return JSON(parse_string()); }
            else if (ch == '[') { return parse_array(); }
            else if (ch == '{') { return parse_object(); }
            else { throw std::invalid_argument("This symbol does not match to any type that we have"); };
        };

    };
};